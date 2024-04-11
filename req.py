import requests

def send_question():
    url = 'http://localhost:5000/submit_question_and_documents'  # Replace with your server URL

    while True:
        question = input("Enter your new question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        data = {
            "question": f"{question} in JSON format. Use a dictionary with key 'question' which contains the question asked, key 'factsByDay' which is a nested JSON object with keys as dates ('yyyy-mm-dd') from the context and values as lists containing all the relevant facts corresponding to each date in response to the question asked",
            "documents": [
                "https://python.langchain.com/docs/expression_language/get_started/",
                "https://www.python.org/about/gettingstarted/",
            ],
            "autoApprove": True
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, json=data, headers=headers)
        
if __name__ == "__main__":
    send_question()
