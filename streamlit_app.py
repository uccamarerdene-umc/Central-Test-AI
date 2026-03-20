# -*- coding: utf-8 -*-
import streamlit as st
import os
import unicodedata
from uuid import uuid4
import sys

# 🔥 Системийн default encoding-ийг UTF-8 болгох (Чухал!)
import importlib
importlib.reload(sys)

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Илүү найдвартай Loader-ууд
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader
)

from langchain.schema import Document as LCDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# ENV & CONFIG
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Индексийн нэр (OpenAI text-embedding-3-small бол 1536 dimension)
index_name = "centralai-v2" 

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API keys are missing! Check your secrets.")
    st.stop()

# ==============================
# UNICODE SAFE CLEANER
# ==============================
def clean_text(text):
    if not text:
        return ""
    # Юникод тэмдэгтүүдийг хэвийн болгох
    text = unicodedata.normalize("NFKC", text)
    # ASCII-р алдаа заадаг тэмдэгтүүдийг аюулгүй хэлбэрт шилжүүлэх
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    # Текст доторх "сонин" тэмдэгтүүдийг цэвэрлэх боловч монгол үсгийг үлдээх
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')

# ==============================
# DOCX LOADER (MONGOLIAN SAFE)
# ==============================
def load_docx_safe(file_path):
    try:
        # Unstructured нь монгол хэл болон тусгай тэмдэгт дээр илүү сайн
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
        docs = loader.load()
        
        safe_docs = []
        for d in docs:
            cleaned_content = clean_text(d.page_content)
            if cleaned_content.strip():
                safe_docs.append(
                    LCDocument(
                        page_content=cleaned_content,
                        metadata={"source": os.path.basename(file_path)}
                    )
                )
        return safe_docs
    except Exception as e:
        st.warning(f"⚠️ Docx Error ({os.path.basename(file_path)}): {str(e)}")
        return []

# ==============================
# LOAD ALL FILES
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
                    docs.extend(load_docx_safe(path))
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                    pdf_docs = loader.load()
                    for d in pdf_docs:
                        d.page_content = clean_text(d.page_content)
                    docs.extend(pdf_docs)
                elif file.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                    txt_docs = loader.load()
                    for d in txt_docs:
                        d.page_content = clean_text(d.page_content)
                    docs.extend(txt_docs)
                elif file.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(path)
                    ppt_docs = loader.load()
                    for d in ppt_docs:
                        d.page_content = clean_text(d.page_content)
                    docs.extend(ppt_docs)
            except Exception as e:
                st.warning(f"⚠️ Skip file {file}: {str(e)}")
    return docs

# ==============================
# MODELS
# ==============================
@st.cache_resource
def init_models():
    # OpenAI Embedding (1536 dims)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Индекс байхгүй бол үүсгэх
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
# SIDEBAR SYNC
# ==============================
with st.sidebar:
    st.header("⚙️ Data Management")
    if st.button("🔄 Sync Documents to Pinecone"):
        with st.spinner("Processing documents..."):
            all_docs = load_all_documents()
            if not all_docs:
                st.error("❌ 'data' хавтас хоосон эсвэл файл олдсонгүй.")
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(all_docs)
                
                # Pinecone Store
                vectorstore = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                
                # Багцаар нь оруулах (Batching)
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    vectorstore.add_documents(batch)
                
                st.success(f"✅ Амжилттай! {len(chunks)} хэсэг мэдээлэл хадгалагдлаа.")

# ==============================
# CHAT INTERFACE
# ==============================
st.title("🤖 Central Test Knowledge Bot")

query = st.text_input("Асуултаа бичнэ үү:", placeholder="Central Test-ийн талаар асуух...")

if query:
    with st.spinner("Хариулт бэлдэж байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )
            
            # Semantic Search
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
                Хэрэв мэдээлэл дотор хариулт байхгүй бол өөрөө зохиож хариулж болохгүй.

                Мэдээлэл:
                {context_text}

                Асуулт: {query}
                """
                
                response = llm.invoke(prompt)
                
                st.markdown("### 🤖 Хариулт:")
                st.write(response.content)
                
                with st.expander("Ашигласан эх сурвалж (Context)"):
                    for doc in search_results:
                        st.info(f"Source: {doc.metadata.get('source')}\n\n{doc.page_content}")
        
        except Exception as e:
            st.error(f"❌ Алдаа гарлаа: {str(e)}")
