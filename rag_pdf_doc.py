import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

st.set_page_config(
    page_title="📄RAG Q&A with PDF & Chat History",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG Q&A with PDF upload & chat history")
st.sidebar.header("👨🏻‍🔧 Configuration")
st.sidebar.write(
    "- Enter your GROQ API key \n"
    "- Upload PDFs \n"
    "- Ask questions and see chat history"
)

# API keys
api_key = st.sidebar.text_input("Groq API Key", type="password")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

uploaded_files = st.file_uploader(
    "📓 🗒 Choose PDF files(s)",
    type="pdf",
    accept_multiple_files=True
)

# Initialize session_state for vector store and chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

# Function to get chat history
def get_history(session_id: str):
    if session_id not in st.session_state.chathistory:
        st.session_state.chathistory[session_id] = ChatMessageHistory()
    return st.session_state.chathistory[session_id]

if uploaded_files:
    with st.spinner("🔄 Loading and splitting PDFs..."):
        all_docs = []
        for pdf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(all_docs)

        # Build FAISS vectorstore fresh every time PDFs uploaded
        st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = st.session_state.vectorstore.as_retriever()

    # Build history-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, decide what to retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Use the retrieved context to answer."
                    "If you don't know, say so. Keep it under three sentences. \n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat UI
    session_id = st.text_input("🆔Session ID", value="default_session")
    user_question = st.chat_input("✍️ Your question comes here....")

    if user_question:
        history = get_history(session_id)
        result = conversational_rag.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}},
        )
        answer = result["answer"]

        # Display chat messages
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

        # Full chat history
        with st.expander("📖 Full chat history"):
            for msg in history.messages:
                role = getattr(msg, "role", msg.type)
                content = msg.content
                st.write(f"{role.title()}: {content}")
else:
    st.info("ℹ️ Upload one or more PDFs above to begin.")
