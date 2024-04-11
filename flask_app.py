import os
import re
import requests
from langchain_community.document_transformers import Html2TextTransformer
from flask import Flask, request, jsonify, Response, render_template
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough    
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import AsyncHtmlLoader

app = Flask(__name__)

# Set up environment variables
os.environ['OPENAI_API_KEY'] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__eeeb4a55378c4edabedda78ddc42734c"

# Initialize the vector and flag variables
vector = None
first_question = False

def convert_date(url):
    match = re.search(r'(\d{4})(\d{2})(\d{2})', url)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3)
        return f'{year}/{month}/{day}'
    else:
        return " "  

@app.route('/')
def submit():
    return "Please submit question and documents."

@app.route('/favicon.ico')
def ignore_favicon():
    return Response(status=204) 

@app.route('/submit_question_and_documents', methods=['POST', 'GET'])
def submit_question_and_documents():
    global vector, first_question

    data = request.json
    question_api = data.get('question')
    documents = data.get('documents')
    auto_approve = data.get('autoApprove', False)
    
    if vector is None:
        # Initialize vector if it's not already initialized
        vector = create_vector(documents[0])
    else:
        # Add new documents to existing vector
        update_vector(vector, documents[1:])
    
    retriever = vector.as_retriever()
    docs_retriver = vector.similarity_search(question_api)

    # Define chat prompts
    template_generate = """Generate questions and answers based on the following context:
        {context}

        Question Prompt: {question_def}
        """

    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer only from given context.

    {context}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    chain = load_qa_chain(
        OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
    )
    
    prompt_generate = ChatPromptTemplate.from_template(template_generate)

    # Initialize chat models
    model_generate = ChatOpenAI(model_name="gpt-3.5-turbo")

    # Define retrieval chains
    retrieval_chain_generate = {
            "context": retriever, "question_def": RunnablePassthrough()
        } | prompt_generate | model_generate | StrOutputParser()

    
    # # Stream results
    if not first_question:
        #questions_and_answers = list(retrieval_chain_generate.stream("generate questions and answers?"))
        for chunk in retrieval_chain_generate.stream("generate questions and answers?"):
            print(chunk, end="", flush=True)
        first_question = True
            
    print('\n')
            
    res = chain.invoke({"input_documents": docs_retriver, "human_input": question_api}, return_only_outputs=True)
    print(res)
    return jsonify(res)

def create_vector(documents):
    date_from_url = convert_date(documents)
    # loader = TextLoader(file_path=documents)
    loader = WebBaseLoader(documents)
    docs = loader.load()
    docs[0].page_content += date_from_url
    split_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vector = FAISS.from_documents(split_documents, OpenAIEmbeddings())
    #vector.add_documents(split_documents)
    return vector

def update_vector(vector, documents):
    for document in documents:
        date_from_url = convert_date(document)
        loader = WebBaseLoader(documents)
        #loader = TextLoader(file_path=document)
        docs = loader.load()
        docs[0].page_content += date_from_url
        split_documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        vector.add_documents(split_documents)

if __name__ == '__main__':
    app.run(debug=True)
