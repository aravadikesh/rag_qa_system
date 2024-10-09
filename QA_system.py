import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
import numpy as np
import faiss
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QASystem:
    def __init__(self, knowledge_path, questions_path):
        self.knowledge_path = knowledge_path
        self.questions_path = questions_path
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.index = None
        self.pipe = None
        self.embedding_model = None
        self.embedding_tokenizer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_qa_pairs()
        self._initialize_models()

    def _load_qa_pairs(self):
        logging.info("Loading Q&A pairs...")
        self.questions, self.answers = self._read_qa_pairs(self.knowledge_path)
        logging.info("Q&A pairs loaded successfully.")

    def _read_qa_pairs(self, file_path):
        questions, answers = [], []
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    question, answer = line.strip().split('\t')
                    questions.append(question)
                    answers.append(answer)
        except Exception as e:
            logging.error(f"Error reading Q&A pairs from {file_path}: {e}")
            raise
        return questions, answers

    def _initialize_models(self):
        logging.info("Loading embedding model and tokenizer...")
        embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(self.device)
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        # self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        logging.info("Embedding model and tokenizer loaded successfully.")

        logging.info("Generating embeddings for questions...")
        preprocessed_questions = self._preprocess_questions(self.questions)
        self.question_embeddings = self._get_embeddings(preprocessed_questions)
        logging.info("Embeddings generated successfully.")

        logging.info("Creating FAISS index...")
        self.index = self._create_faiss_index(self.question_embeddings)
        logging.info("FAISS index created successfully.")

        logging.info("Loading text generation model and tokenizer...")
        model_id = "microsoft/Phi-3-mini-128k-instruct"
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU mode
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        logging.info("Text generation model and tokenizer loaded successfully.")

    def _preprocess_text(self, text):
        return text.lower()

    def _preprocess_questions(self, questions):
        return [self._preprocess_text(q) for q in questions]

    # def _get_embeddings(self, texts):
    #     inputs = self.embedding_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    #     with torch.no_grad():
    #         outputs = self.embedding_model(**inputs)
    #     return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def _get_embeddings(self, texts):
        inputs = self.embedding_tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  

    def _create_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def generate_unique_answer(self, prompt, top_k=3, max_new_tokens=200):
        try:
            preprocessed_prompt = self._preprocess_text(prompt)
            prompt_embedding = self._get_embeddings([preprocessed_prompt])[0]
            _, I = self.index.search(np.array([prompt_embedding]), top_k)
            similar_answers = [self.answers[idx] for idx in I[0]]
            combined_answers = " ".join(similar_answers)

            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": combined_answers}
            ]

            output = self.pipe(messages, max_new_tokens=max_new_tokens, return_full_text=False, temperature=0.5, do_sample=True)
            generated_answer = output[0]['generated_text']
            return generated_answer
        except Exception as e:
            logging.error(f"Error generating unique answer for prompt '{prompt}': {e}")
            raise

    def generate_answers(self):
        qa_pairs = []
        with open(self.questions_path, 'r') as q_file:
            user_questions = q_file.readlines()
        
        batch_size = 5

        for i in range(0, len(user_questions), batch_size):
            batch = user_questions[i:i + batch_size]
            prompts = [q.strip() for q in batch]
            
            start_time = time.time()
            unique_answers = self._generate_unique_answers(prompts)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            print(f"Time taken for batch: {elapsed_time:.2f} seconds")
            
            for prompt, unique_answer in zip(prompts, unique_answers):
                qa_pairs.append({"question": prompt, "answer": unique_answer})

        return qa_pairs

    def _generate_unique_answers(self, prompts):
        unique_answers = []
        prompt_embeddings = self._get_embeddings(prompts)

        for prompt, prompt_embedding in zip(prompts, prompt_embeddings):
            D, I = self.index.search(prompt_embedding.reshape(1, -1), k=2)
            closest_question_index = I[0][0]
            closest_answer = self.answers[closest_question_index]
            generated_answer = self.pipe(f"{prompt} {closest_answer}", max_length=150, num_return_sequences=1)[0]['generated_text']
            unique_answers.append(generated_answer)

        return unique_answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the QA system.")
    parser.add_argument('--knowledge_path', type=str, required=True, help="Path to the knowledge file with Q&A pairs")
    parser.add_argument('--questions_path', type=str, required=True, help="Path to the file with user questions")
    parser.add_argument('--prompt', type=str, required=False, help="Prompt to generate an answer (optional)")
    
    args = parser.parse_args()

    # Initialize the system
    qa_system = QASystem(knowledge_path=args.knowledge_path, questions_path=args.questions_path)
    
    if args.prompt:
        unique_answer = qa_system.generate_unique_answer(args.prompt)
        print(f"Unique answer for prompt '{args.prompt}': {unique_answer}")
    else:
        qa_pairs = qa_system.generate_answers()

        for qa in qa_pairs:
            print(f"Q: {qa['question']}\nA: {qa['answer']}\n")
