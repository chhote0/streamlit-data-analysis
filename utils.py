from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


genai.configure(api_key="AIzaSyBRkhTYU6Is6Q_Yvkrd_s8v_caZrcUK4nM")
os.environ["GOOGLE_API_KEY"]="AIzaSyBRkhTYU6Is6Q_Yvkrd_s8v_caZrcUK4nM"


# Function to get response from GEMINI PRO
def get_model_response(file, query):

    # Split the context text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in file)

    data = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    searcher = Chroma.from_texts(data, embeddings).as_retriever()

    q = "Which employee has maximum salary?"
    records = searcher.get_relevant_documents(q)
    print(records)