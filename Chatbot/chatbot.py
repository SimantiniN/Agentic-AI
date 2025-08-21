import os
import certifi
from dotenv import load_dotenv
import streamlit as st
import json
import re


# ---- Load Environment Variables ----
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FILE_PATH = os.getenv("FILE_PATH")
IMAGE_PATH  = os.getenv("IMAGE_PATH")
# ---- LangChain Imports ----
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate

# ---- LLM and Embeddings ----
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# ---- Load PDF and Split ----
loader = PyPDFLoader(FILE_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# ---- Create Vectorstore and Retriever ----
vectorstore = FAISS.from_documents(chunks, embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---- Prompt Template with JSON Output ----
system_template = """
You are a helpful assistant. Answer the question in English and translate it into {lang_choice}.
Return the output as JSON in the following format:

{{
  "english_answer": "...",
  "{lang_choice}_answer": "..."
}}

Context:
{context}

Question:
{query}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{query}")
])

# ---- LLMChain ----
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# ---- Streamlit UI ----
# Display image of Shivaji Maharaj
if IMAGE_PATH and os.path.exists(IMAGE_PATH):
    col1, col2 = st.columns([1,1])  # adjust ratios as needed
    with col1:
        st.image(IMAGE_PATH, width=300) 
    with col2:
        st.markdown("<div style='padding-top:200px;'>", unsafe_allow_html=True)
        st.title("🔎 Shivaji Maharaj Q&A")
        st.markdown("</div>", unsafe_allow_html=True)
        #st.title("🔎 Shivaji Maharaj Q&A ") # small image next to the title

    # st.image(IMAGE_PATH, caption="Shivaji Maharaj", width=300)
    # st.title("🔎 Shivaji Maharaj Q&A.")
else:
    st.warning("Image not found. Please check IMAGE_PATH in your .env file.")

st.write("Ask questions about Shivaji Maharaj from the uploaded PDF.")

query = st.text_input("Enter your question:")
lang_choice = st.selectbox("Show answer in:",["English","Marathi", "Hindi", "Spanish", "French", "German", "Japanese", "Chinese", "Tamil", "Telugu", "Kannada", "Gujarati", "Bengali", "Punjabi", "Arabic", "Russian"])


if query:
    # ---- Retrieve relevant docs ----
    relevant_docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    # ---- Run LLMChain ----
    prompt_vars = {
        "query": query,
        "lang_choice": lang_choice,
        "context": context_text
    }
    raw_output = llm_chain.run(**prompt_vars)

    # ---- Parse JSON safely ----
    # try:
    #     result_json = json.loads(raw_output)
    #     english_answer = result_json.get("english_answer", "Not found")
    #     translated_answer = result_json.get(f"{lang_choice}_answer", "Not found")
    # except json.JSONDecodeError:
    #     english_answer = raw_output
    #     translated_answer = "Translation not detected"


# ---- Safely parse JSON from LLM output ----
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if match:
        try:
            result_json = json.loads(match.group())
            english_answer = result_json.get("english_answer", "Not found")
            translated_answer = result_json.get(f"{lang_choice}_answer", "Not found")
        except json.JSONDecodeError:
            english_answer = raw_output
            translated_answer = "Translation not detected"
    else:
        english_answer = raw_output
        translated_answer = "Translation not detected"

    # ---- Display Answers ----
    if lang_choice == "English":
        st.subheader("Answer (English):")
        st.markdown(f"<p style='color:blue; font-size:20px;'>{english_answer}</p>", unsafe_allow_html=True)
    else :
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("English Answer:")
            st.markdown(f"<p style='color:blue; font-size:18px;'>{english_answer}</p>", unsafe_allow_html=True)
        with col2:
            st.subheader(f"{lang_choice} Answer:")
            st.markdown(f"<p style='color:green; font-size:18px;'>{translated_answer}</p>", unsafe_allow_html=True)

    # ---- Display Sources ----
    with st.expander("📖 Sources"):
        for i, doc in enumerate(relevant_docs, 1):
            metadata = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items())
            st.markdown(f"**Source {i}:** {metadata}")
            st.write(doc.page_content[:300] + "...")
