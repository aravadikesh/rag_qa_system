import argparse
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ResponseModel, TextResult
from QA_system import QASystem

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run the QA System Server")
    parser.add_argument('--knowledge', type=str, required=True, help='Path to the knowledge file')
    parser.add_argument('--questions', type=str, required=True, help='Path to the questions file')
    return parser.parse_args()

# Create a server
server = MLServer(__name__)

# Initialize the QASystem with command-line arguments
args = parse_args()
model = QASystem(args.knowledge, args.questions)

# Create an endpoint
@server.route("/qasystem", DataTypes.TEXT)
def process_text(inputs: list, parameters: dict) -> dict:
    # questions = [e["text"] for e in inputs
    res = model.generate_unique_answer("What is NumPy")  # Use the model to generate answers
    print(res)
    res2 = [TextResult(text=res)]
    response = ResponseModel(results=res2)
    return response.get_response()

if __name__ == '__main__':
    server.run()
