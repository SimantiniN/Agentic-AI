import os
import certifi  # type: ignore
from dotenv import load_dotenv  # type: ignore
import streamlit as st  # type: ignore

# ---- Load Environment Variables ----
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FILE_PATH = os.environ.get("FILE_PATH")

# ---- LangChain Imports ----
from langchain_groq import ChatGroq  #  using Groq API
from langchain_ollama import OllamaEmbeddings  # still use Ollama embeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ---- LLM (Groq) ----
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="gemma2-9b-it"  # choose Groq-supported model
)

# ---- Embeddings ----
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# ---- Load PDF ----
file_path = FILE_PATH
loader = PyPDFLoader(file_path)
docs = loader.load()

# ---- Split into Chunks ----
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)

# ---- Create Vectorstore ----
vectorstore = FAISS.from_documents(final_documents, embeddings_model)

# ---- Retriever ----
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- Retrieval QA ----
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

# ---- Streamlit UI ----
st.title("🔎 Shivaji Maharaj Q&A with Chatbot (Groq-Powered)")
st.write("Ask questions about Shivaji Maharaj (from the uploaded PDF).")

query = st.text_input("Enter your question:")
lang_choice = st.selectbox("Show answer in:", ["English", "Marathi", "Hindi"])

if query:
    # Prompt with structured format
    if lang_choice == "English":
        prompt = f"""Answer the following question in English only:
        Question: {query}
        """
    else:
        prompt = f"""Answer the following question in English, 
        then provide translation in {lang_choice}.
        
        Format:
        English Answer:
        <text>
        {lang_choice} Answer:
        <text>
        
        Question: {query}
        """

    answer = qa.invoke({"query": prompt})
    result_text = answer["result"]

    # Parsing
    if lang_choice == "English":
        st.subheader("Answer (English):")
        st.markdown(
            f"<p style='color:blue; font-size:20px; font-family:Arial;'>{result_text}</p>",
            unsafe_allow_html=True
        )
    else:
        parts = result_text.split(f"{lang_choice} Answer:")
        english_part = parts[0].replace("English Answer:", "").strip()
        translated_part = parts[1].strip() if len(parts) > 1 else " Translation not detected."

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("English Answer:")
            st.markdown(f"<p style='color:blue; font-size:18px;'>{english_part}</p>", unsafe_allow_html=True)
        with col2:
            st.subheader(f"{lang_choice} Answer:")
            st.markdown(f"<p style='color:green; font-size:18px;'>{translated_part}</p>", unsafe_allow_html=True)

    # Sources
    with st.expander("📖 Sources"):
        for i, doc in enumerate(answer["source_documents"], 1):
            st.markdown(f"**Source {i}:** {doc.metadata}")
            st.write(doc.page_content[:300] + "...")
