# -*- coding: utf-8 -*-
import streamlit as st
import os
import unicodedata
from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Найдвартай, хөнгөн Loader-ууд
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader
)

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# ENV & CONFIG
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

# API Keys
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Index name (OpenAI text-embedding-3-small = 1536 dimensions)
index_name = "centralai-v2" 

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API keys are missing! Please check Streamlit Secrets.")
    st.stop()

# ==============================
# UNICODE SAFE CLEANER
# ==============================
def clean_text(text):
    if not text:
        return ""
    # Юникод тэмдэгтүүдийг хэвийн болгох (NFKC)
    text = unicodedata.normalize("NFKC", text)
    # ASCII алдаа үүсгэдэг тусгай тэмдэгтүүдийг солих
    replacements = {
        '\u2013': '-', '\u2014': '-', 
        '\u2018': "'", '\u2019': "'", 
        '\u201c': '"', '\u201d': '"',
        '\u00a0': ' '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Монгол үсэг болон стандарт тэмдэгтүүдийг үлдээх
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

# ==============================
# MODELS
# ==============================
@st.cache_resource
def init_models():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Индекс байхгүй бол үүсгэх (Dimension 1536)
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return embeddings

embeddings = init_models()

# ==============================
# DATA LOADING
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
                st.warning(f"⚠️ Алдаа ({file}): {str(e)}")
    return docs

# ==============================
# SIDEBAR SYNC
# ==============================
with st.sidebar:
    st.header("⚙️ Data Management")
    if st.button("🔄 Sync Documents to Pinecone"):
        with st.spinner("Processing..."):
            all_docs = load_all_documents()
            if not all_docs:
                st.error("❌ Файл олдсонгүй.")
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
                chunks = splitter.split_documents(all_docs)
                
                vectorstore = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                
                # Batch upload
                batch_size = 80
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    vectorstore.add_documents(batch)
                
                st.success(f"✅ Амжилттай! {len(chunks)} хэсэг мэдээлэл хадгалагдлаа.")

# ==============================
# CHAT INTERFACE
# ==============================
st.title("🤖 Central Test AI Assistant")

query = st.text_input("Асуултаа бичнэ үү:", placeholder="Мэдээллийн сангаас хайх...")

if query:
    with st.spinner("AI хариулт бэлдэж байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )
            
            # Semantic search
            search_results = vectorstore.similarity_search(query, k=5)
            
            if not search_results:
                st.warning("⚠️ Холбогдох мэдээлэл олдсонгүй.")
            else:
                context_text = "\n\n".join([doc.page_content for doc in search_results])
                
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.1
                )
                
                prompt = f"""
                Та бол Central Test компанийн туслах AI байна. 
                Доорх мэдээлэлд (Context) тулгуурлан асуултанд монгол хэлээр маш тодорхой хариул.
                Хариулт мэдээлэл дотор байхгүй бол 'Мэдээлэл алга байна' гэж хэлнэ үү.

                Мэдээлэл:
                {context_text}

                Асуулт: {query}
                """
                
                response = llm.invoke(prompt)
                st.markdown("### 🤖 Хариулт:")
                st.write(response.content)
                
                with st.expander("Эх сурвалж харах"):
                    for doc in search_results:
                        st.info(f"Source: {doc.metadata.get('source')}\n\n{doc.page_content}")
        
        except Exception as e:
            st.error(f"❌ Алдаа: {str(e)}")
