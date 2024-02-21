import os
import warnings
import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain import hub
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

warnings.filterwarnings("ignore")
load_dotenv()

# Create streamlit page
st.title("Chat with WEPA Data Bot")
st.image("logo.png")

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

db = FAISS.load_local("vectorstore", embeddings)
retriever = db.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = AzureChatOpenAI(
    model="gpt35turbo",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

prompt = hub.pull("rlm/rag-prompt")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_message = st.text_input("Please enter your message ")
if user_message:
    response = rag_chain.invoke(user_message)

    st.markdown(response)