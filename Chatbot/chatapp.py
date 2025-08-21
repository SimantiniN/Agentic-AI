import os
import certifi
from dotenv import load_dotenv
import streamlit as st

# ---- Load Environment Variables ----
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FILE_PATH = os.getenv("FILE_PATH")

# ---- LangChain Imports ----
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate

# ---- Initialize LLM and Embeddings ----
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# ---- Load and Split PDF ----
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)

# ---- Vectorstore and Retriever ----
vectorstore = FAISS.from_documents(final_documents, embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- Prompt Template ----
system_template = "Answer the following question in English, then provide translation into {lang_choice}."
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{query}")
])

# ---- LLMChain ----
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ---- Streamlit UI ----
st.title("🔎 Shivaji Maharaj Q&A Chatbot (Groq-Powered)")
st.write("Ask questions about Shivaji Maharaj (from the uploaded PDF).")

query = st.text_input("Enter your question:")
lang_choice = st.selectbox("Show answer in:", ["English", "Marathi", "Hindi"])

if query:
    # ---- Get documents from retriever ----
    docs = retriever.get_relevant_documents(query)
    combined_text = "\n\n".join([doc.page_content for doc in docs])

    # ---- Run LLMChain with ChatPromptTemplate ----
    prompt_vars = {
        "query": f"{query}\n\nContext:\n{combined_text}",
        "lang_choice": lang_choice
    }
    raw_output = llm_chain.run(**prompt_vars)

    # ---- Parse output ----
    if lang_choice == "English":
        st.subheader("Answer (English):")
        st.markdown(f"<p style='color:blue; font-size:20px;'>{raw_output}</p>", unsafe_allow_html=True)
    else:
        parts = raw_output.split(f"{lang_choice} Answer:")
        english_part = parts[0].replace("English Answer:", "").strip()
        translated_part = parts[1].strip() if len(parts) > 1 else "Translation not detected."

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("English Answer:")
            st.markdown(f"<p style='color:blue; font-size:18px;'>{english_part}</p>", unsafe_allow_html=True)
        with col2:
            st.subheader(f"{lang_choice} Answer:")
            st.markdown(f"<p style='color:green; font-size:18px;'>{translated_part}</p>", unsafe_allow_html=True)

    # ---- Display sources ----
    with st.expander("📖 Sources"):
        for i, doc in enumerate(docs, 1):
            metadata = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
            st.markdown(f"**Source {i}:** {metadata}")
            st.write(doc.page_content[:300] + "...")
