# -*- coding: utf-8 -*-
import streamlit as st
import os
import unicodedata
from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Найдвартай Loader-ууд
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader
)

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# API KEY SAFETY CLEANER
# ==============================
def get_safe_secret(key):
    """API Key-д байж болох Unicode тэмдэгтүүдийг цэвэрлэж ASCII болгоно."""
    val = st.secrets.get(key)
    if val:
        # Аливаа үл үзэгдэх Unicode тэмдэгтүүдийг устгаж ASCII болгох
        return str(val).encode("ascii", "ignore").decode("ascii").strip()
    return None

# ==============================
# ENV & CONFIG
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

# API түлхүүрүүдийг аюулгүйгээр авах
google_api_key = get_safe_secret("GOOGLE_API_KEY")
pinecone_api_key = get_safe_secret("PINECONE_API_KEY")
openai_api_key = get_safe_secret("OPENAI_API_KEY")

index_name = "centralai-v2" 

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API keys are missing in Streamlit Secrets! Secrets хэсгээ шалгана уу.")
    st.stop()

# ==============================
# UNICODE / ASCII SAFE CLEANER
# ==============================
def clean_text(text):
    if not text:
        return ""
    # Юникод тэмдэгтүүдийг хэвийн болгох
    text = unicodedata.normalize("NFKC", text)
    # ASCII-д алдаа заадаг тусгай тэмдэгтүүдийг гараар солих
    replacements = {
        '\u2013': '-', '\u2014': '-', 
        '\u2018': "'", '\u2019': "'", 
        '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u00a0': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Монгол үсгийг оролцуулан хэвлэгдэх боломжтой тэмдэгтүүдийг үлдээх
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

# ==============================
# MODELS INITIALIZATION
# ==============================
@st.cache_resource
def init_models():
    # OpenAI Embedding (1536 dim)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    # Pinecone Client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Индекс шалгах / үүсгэх
    try:
        existing_indexes = [i["name"] for i in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
    except Exception as e:
        st.error(f"Pinecone Index Error: {str(e)}")
        
    return embeddings

embeddings = init_models()

# ==============================
# DOCUMENT LOADER
# ==============================
def load_all_documents():
    docs = []
    data_dir = "data"
    if not os.path.exists(data_dir):
        return docs

    for root, _, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".docx"):
                    loader = Docx2txtLoader(path)
                    raw_docs = loader.load()
                    for d in raw_docs:
                        content = clean_text(d.page_content)
                        if content.strip():
                            docs.append(LCDocument(page_content=content, metadata={"source": file}))
                
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    raw_docs = loader.load()
                    for d in raw_docs:
                        d.page_content = clean_text(d.page_content)
                        docs.append(d)
                
                elif file.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                    raw_docs = loader.load()
                    for d in raw_docs:
                        d.page_content = clean_text(d.page_content)
                        docs.append(d)
                        
            except Exception as e:
                st.warning(f"⚠️ Алдаа гарлаа ({file}): {str(e)}")
    return docs

# ==============================
# UI & SYNC
# ==============================
st.title("🤖 Central Test AI Assistant")

with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Documents"):
        with st.spinner("Мэдээллийг боловсруулж байна..."):
            all_docs = load_all_documents()
            if not all_docs:
                st.error("❌ 'data' хавтсанд файл олдсонгүй.")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_documents(all_docs)
                
                try:
                    # Pinecone-д холбогдох
                    vectorstore = PineconeVectorStore(
                        index_name=index_name,
                        embedding=embeddings,
