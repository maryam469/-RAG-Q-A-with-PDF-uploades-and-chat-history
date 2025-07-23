
# How to load multiple PDFs into vector database
# how to split large documents into small chunks for better embeddings
# how to use retrieval augmented generation (RAG) with langchain chain that combines a vector stroe retriever + an llm (GROQ) + Embedding (Hugging Face)+ a prompt template + conversational Q&A chat + Unique Session ID wise

# Process from upload till extraction
# Load PDF file -> Convert their contents into vector embeddings ->  implemented a chat history so that each conversationn is remembered -> how the user session logic ( with session_id) helps each user maintain their own converstion flow


import os
import time         ##for anytime/debug stamps (if needed)
import tempfile    ## to store and uploads PGFS on disk temporarily
import streamlit as st
from dotenv import load_dotenv

##Langchain core classes and utilitties
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

##langchain  LLM and chaining utitlities
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain   ##Agar aap ke paas multiple documents hain (e.g., PDFs, notes, web pages), aur aap un sab ki combined knowledge se ek answer lena chahte hain, to create_stuff_documents_chain use hota hai.
from langchain.chains import create_history_aware_retriever, create_retrieval_chain    ##create_history_aware_retriever previous conversation context ko yad rkh kr relevant document retrieve kr ske.  ##RAG chain bnata ha jis main user ki query k sath documents reterive kr ke unka answer generate hota ha.

##text sppliting and embeedings
from langchain.text_splitter import RecursiveCharacterTextSplitter   ##ye langchain ka text splitter tool hai jo text ya documents ko chote chunks main torta ha  take unhain easily LLM ke context ke window main fit kia ja ske.
from langchain.embeddings import HuggingFaceEmbeddings


##vector store 
from langchain.vectorstores import Chroma  #vectorstore aik database hota ha jo document ko embbedings main convert kr ke store krta ha.##Chroma Hum  tab use karte hain jab hume documents ko vector form me store karke RAG chatbot, similarity-based search, ya lightweight open-source vector DB chahiye hoti hai.


##PDF file loader (loads a single pdf into docs)

from langchain_community.document_loaders import PyPDFLoader

##Load environment variables 

load_dotenv()

#streamlit page setup

st.set_page_config(
    page_title="📄RAG Q&A with PDF & Chat History",
    layout="wide",
    initial_sidebar_state="expanded"  ## ye setting ensure karti hai ke sidebar start hote hi open  dikhe,
)


st.title ("📄 RAG Q&A with PDF uploades and chat history")
st.sidebar.header("👨🏻‍🔧 Configuration")
st.sidebar.write(
    "- Enter your GROQ API key \n"
    "- Uploads PDFS on the main page \n"
    "- Ask questions and see chat history"
)

#API keys # embedding setup
api_key = st.sidebar.text_input("Groq API Key", type="password")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")  #Ye line Python me environment variable set kar rahi hai









embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


#only proceed if the user has entered their Groq key

if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

##instantiate the GROQ LLM
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")


##file uploader : allow multiple PDF UPLOADS

uploaded_files = st.file_uploader(
    "📓 🗒 Choose PDF files(s)",
    type="pdf",
    accept_multiple_files=True,   ##Ek se zyada files select/upload karne dega
) 

##A placeholder to collect all documents
all_docs = []

if uploaded_files:
    ##show progress spinner
    with st.spinner("🔄 Loading and splitting PDFs "):
        for pdf in uploaded_files:
            ##write to a temp file so pyPDFloader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name


            #load the pdf into a list of documment objects
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
            

##split docs into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 150,
    )
    splits = text_splitter.split_documents(all_docs) #ye line ap ke multiple documents ko chote chote chunks main divide krti ha using text_splitter taake ap unhain later process embed ya reterive kr sko.


##Build or load the chroma vector store (chaching for performance)

    @st.cache_resource(show_spinner=False)  ##streamlit decorator ha jo function ko cache krta ha.
    def get_vectorstore(_splits):   
        return Chroma.from_documents(
            _splits,     #chote document chunks
            embeddings,  #ap ka predefined model Huggingface and Groq 
            persist_directory= "./chroma_index"  #directory jahan vector DB store hogi.
        )
    vectorstore = get_vectorstore(splits)   #ye function ko call krta ha splits k sath.
    retriever = vectorstore.as_retriever()   ##ye line chroma vectorstotre ko reteriver main convert krta ha jisse ap similar document search kr sko user query ki base pr.

##Build a history-aware reteriver that uses past chat to refine searched.

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
    ##QA chain "stuff" all retrieved docs into the LLM
    qa_prompt = ChatPromptTemplate.from_messages([  ##ye langchain ka method ha jo multi message chat prompt bnata ha(jaise system, user, assistant)
        ("system", "You are an assistant. Use the retrieved context to answer."
                    "If you don't know, say so. Keep it under three sentences. \n\n{context}"),
        MessagesPlaceholder("chat_history"),  ##ye previous conversation ko 
        ("human", "{input}"),  ##ye user ka current question placeholder hai
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)  #yelangchain ka built in method ha jo document+prompt ko llm ko send krta ha.
    rag_chain=create_retrieval_chain(history_aware_retriever, question_answer_chain)  ##create reterival chain langchain ka function ha jo user input aur chat history le kr history_aware reteriver ko query bnae deta ha then question_answer_chain ko pass krta ha for final answer.


    #session state for chat history
    if "chathistory" not in st.session_state:
        st.session_state.chathistory={}

    def get_history(session_id: str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]


###wraap the ragchain so ti automatically logs history

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

##chat UI 
    session_id = st.text_input("🆔Sesssion ID", value="default_session")
    user_question = st.chat_input("✍️ Your question comes here....")


    if user_question:
        history = get_history (session_id)
        result = conversational_rag.invoke(
            {"input" : user_question},
            config={"configurable" : {"session_id": session_id}},
        )
        answer = result["answer"]
    

        #display in streamlit new chat format
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)
    
        with st.expander("📖 Full chat history"):
            for msg in history.messages:
                # msg rolw is typically "human" or "assistant"
                role = getattr(msg, "role", msg.type)
                content = msg.content
                st.write(f" {role.title()}: {content}")
else:
    # No file is uploaded yet
    st.info("ℹ️ Upload one or more PDFs above to begin.")