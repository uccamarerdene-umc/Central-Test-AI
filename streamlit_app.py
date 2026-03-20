# -*- coding: utf-8 -*-
import streamlit as st
import os
import unicodedata
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# CONFIG & SECRETS
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

def get_safe_secret(key_name):
    val = st.secrets.get(key_name)
    if val:
        # Unicode зураас (—) болон илүү зайг цэвэрлэх
        clean_val = str(val).replace('—', '-').strip()
        return clean_val.encode("ascii", "ignore").decode("ascii")
    return None

google_api_key = get_safe_secret("GOOGLE_API_KEY")
pinecone_api_key = get_safe_secret("PINECONE_API_KEY")
openai_api_key = get_safe_secret("OPENAI_API_KEY")

index_name = "centralai-v2"

if not all([google_api_key, pinecone_api_key, openai_api_key]):
    st.error("❌ API Keys are missing in Secrets! Settings -> Secrets хэсэгт түлхүүрүүдээ оруулна уу.")
    st.stop()

# ==============================
# MODELS INITIALIZATION
# ==============================
@st.cache_resource
def init_models():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
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
        st.error(f"Pinecone Init Error: {str(e)}")
    return embeddings

embeddings = init_models()

# ==============================
# DOCUMENT PROCESSING
# ==============================
def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

def load_docs():
    docs = []
    if not os.path.exists("data"): return docs
    for root, _, files in os.walk("data"):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".docx"): loader = Docx2txtLoader(path)
                elif file.endswith(".pdf"): loader = PyPDFLoader(path)
                elif file.endswith(".txt"): loader = TextLoader(path, encoding="utf-8")
                else: continue
                raw = loader.load()
                for d in raw:
                    docs.append(LCDocument(page_content=clean_text(d.page_content), metadata={"source": file}))
            except: continue
    return docs

# ==============================
# UI
# ==============================
st.title("🤖 Central Test AI Assistant")

with st.sidebar:
    if st.button("🔄 Sync Documents"):
        with st.spinner("Processing documents..."):
            documents = load_docs()
            if documents:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(documents)
                vectorstore = PineconeVectorStore.from_documents(
                    chunks, embeddings, index_name=index_name, pinecone_api_key=pinecone_api_key
                )
                st.success("✅ Дата амжилттай хадгалагдлаа!")
            else:
                st.error("❌ 'data' хавтсанд файл олдсонгүй.")

query = st.text_input("Асуултаа бичнэ үү:")
if query:
    with st.spinner("Searching..."):
        try:
            vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
            results = vectorstore.similarity_search(query, k=5)
            if results:
                context = "\n\n".join([doc.page_content for doc in results])
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                response = llm.invoke(f"Дараах мэдээлэлд тулгуурлан хариул: {context}\n\nАсуулт: {query}")
                st.markdown(response.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")
