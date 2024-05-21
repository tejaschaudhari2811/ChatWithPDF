import os
import warnings
import streamlit as st
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory


warnings.filterwarnings("ignore")
load_dotenv()

# Create streamlit page
st.title("Chat with a 300 page book about Data Mesh")
st.image("logos/logo_insights.png")
st.sidebar.image("logos/book.jpg")
st.sidebar.link_button("Go to the book", os.getenv("BOOK_LINK"))

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment="textEmbeddingModel",
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
)

db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = AzureChatOpenAI(
    model="gpt35turbo",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

# Prompt template
template = """ You are a Question Answering bot about a BOOK. You answer in complete sentences and step-by-step whenever necessary. You answer in precise manner.
You always provide references from the book Data Management at Scale by Piethein Strengholt with page number and paragraph start sentence in 
double quotation marks. 
Answer the question based only on the following context, which can include information about Data:
{context}
Question: {input}
"""

# Add history
question_summary_prompt = """Given a chat history and the latest user question \
    which  the context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otheriwse return it as is."""

question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", question_summary_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, question_prompt)


prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt2)

rag_chain_with_hisory = create_retrieval_chain(
    history_aware_retriever, qa_chain)

st.markdown(
    """Enter your text (prompt) in the following box. Always prompt as concrete as possible and \
             always cross check the references in the book that the tool provides. Please provide Feedback to Tejas :blush:."""
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.markdown(message.content)
    else:
        with st.chat_message("CEO"):
            st.markdown(message.content)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_with_hisory,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

question = st.chat_input("Please Enter your interview question..")

if question is not None and question != "":
    st.session_state.chat_history.append(HumanMessage(question))
    with st.chat_message("User"):
        st.markdown(question)

    with st.chat_message("CEO"):
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": "abc123"}
            }
        )["answer"]
        st.markdown(response)
    st.session_state.chat_history.append(AIMessage(response))