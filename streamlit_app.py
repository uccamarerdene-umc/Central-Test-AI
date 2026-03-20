# -*- coding: utf-8 -*-
import streamlit as st
import os
import unicodedata
from uuid import uuid4
from dotenv import load_dotenv

# LangChain & AI сангууд
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# API KEY SAFETY CLEANER
# ==============================
def get_safe_secret(key_name):
    """Secrets-ээс нэрээр нь дуудаж, Unicode алдааг цэвэрлэнэ."""
    val = st.secrets.get(key_name)
    if val:
        # Unicode зураас болон илүү зайг цэвэрлэх
        val = str(val).replace('—', '-').strip()
        return val.encode("ascii", "ignore").decode("ascii")
    return None

# ==============================
# ENV & CONFIG
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

# АНХААР: Энд API Key-г шууд бичиж болохгүй! Зөвхөн Secrets дэх нэрийг нь бичнэ.
google_api_key = get_safe_secret("GOOGLE_API_KEY")
pinecone_api_key = get_safe_secret("PINECONE_API_KEY")
openai_api_key = get_safe_secret("OPENAI_API_KEY")

index_name = "centralai-v2" 

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API Keys missing! Streamlit Cloud-ийн Settings -> Secrets хэсэгт түлхүүрүүдээ оруулна уу.")
    st.stop()

# ==============================
# UNICODE / ASCII SAFE CLEANER
# ==============================
def clean_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        '\u2013': '-', '\u2014': '-', 
        '\u2018': "'", '\u2019': "'", 
        '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u00a0': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

# ==============================
# MODELS INITIALIZATION
# ==============================
@st.cache_resource
def init_models():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
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
        st.error(f"Pinecone Error: {str(e)}")
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
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                elif file.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                else:
                    continue
                raw_docs = loader.load()
                for d in raw_docs:
                    content = clean_text(d.page_content)
                    if content.strip():
                        docs.append(LCDocument(page_content=content, metadata={"source": file}))
            except Exception as e:
                st.warning(f"⚠️ Load error ({file}): {str(e)}")
    return docs

# ==============================
# UI & SYNC
# ==============================
st.title("🤖 Central Test AI Assistant")

with st.sidebar:
    if st.button("🔄 Sync Documents"):
        with st.spinner("Processing..."):
            all_docs = load_all_documents()
            if not all_docs:
                st.error("❌ 'data' хавтсанд файл алга.")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(all_docs)
                try:
                    vectorstore = PineconeVectorStore.from_documents(
                        chunks, 
                        embeddings, 
                        index_name=index_name,
                        pinecone_api_key=pinecone_api_key
                    )
                    st.success("✅ Амжилттай хадгалагдлаа!")
                except Exception as e:
                    st.error(f"Sync Error: {str(e)}")

# ==============================
# CHAT INTERFACE
# ==============================
query = st.text_input("Асуултаа бичнэ үү:")

if query:
    with st.spinner("Searching..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )
            results = vectorstore.similarity_search(query, k=5)
            if results:
                context = "\n\n".join([doc.page_content for doc in results])
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                prompt = f"Дараах мэдээлэлд тулгуурлан хариул: {context}\n\nАсуулт: {query}"
                response = llm.invoke(prompt)
                st.write(response.content)
        except Exception as e:
            st.error(f"Chat Error: {str(e)}")
