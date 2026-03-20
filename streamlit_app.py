import streamlit as st
import os
import unicodedata
from uuid import uuid4

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    DirectoryLoader
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# 1. ENV
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

index_name = "centralai-embed-3-small"
dimension = 1536

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API түлхүүр дутуу байна!")
    st.stop()

# ==============================
# 2. TEXT CLEAN (🔥 FULL SAFE)
# ==============================
def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("—", "-")
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text

# ==============================
# 3. LOAD FILES
# ==============================
def load_all_documents():
    docs = []

    loaders = [
        ("data", "**/*.docx", Docx2txtLoader),
        ("data", "**/*.pdf", PyPDFLoader),
        ("data", "**/*.txt", TextLoader),
        ("data", "**/*.pptx", UnstructuredPowerPointLoader),
    ]

    for path, pattern, loader_cls in loaders:
        if os.path.exists(path):
            try:
                loader = DirectoryLoader(
                    path,
                    glob=pattern,
                    loader_cls=loader_cls,
                    show_progress=True
                )
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"⚠️ Loader алдаа ({pattern}): {e}")

    return docs

# ==============================
# 4. MODELS
# ==============================
@st.cache_resource
def load_models():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    pc = Pinecone(api_key=pinecone_api_key)

    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return embeddings

embeddings = load_models()

# ==============================
# 5. SYNC (🔥 FIXED)
# ==============================
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Sync Data to Pinecone"):
        if not os.path.exists("data"):
            st.error("❌ data хавтас олдсонгүй")
        else:
            with st.spinner("📤 Sync хийж байна..."):
                try:
                    docs = load_all_documents()

                    if not docs:
                        st.warning("⚠️ файл олдсонгүй")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=120,
                            separators=["\n\n", "\n", ".", "!", "?", " "]
                        )

                        texts = splitter.split_documents(docs)

                        vectorstore = PineconeVectorStore(
                            index_name=index_name,
                            embedding=embeddings,
                            pinecone_api_key=pinecone_api_key
                        )

                        safe_texts = []
                        metadatas = []
                        ids = []

                        for doc in texts:
                            content = clean_text(doc.page_content)

                            # 🔥 ASCII SAFE
                            content = content.encode("ascii", "ignore").decode()

                            meta = {}
                            if doc.metadata:
                                for k, v in doc.metadata.items():
                                    meta[clean_text(k)] = clean_text(v)

                            safe_texts.append(content)
                            metadatas.append(meta)
                            ids.append(str(uuid4()))

                        vectorstore.add_texts(
                            texts=safe_texts,
                            metadatas=metadatas,
                            ids=ids
                        )

                        st.success(f"✅ {len(safe_texts)} chunk хадгалагдлаа")

                except Exception as e:
                    st.error(f"❌ Sync алдаа: {str(e)}")

# ==============================
# 6. QUERY
# ==============================
def enhance_query(query):
    return f"Монгол хэл дээр ойлгомжтой хайлт: {query}"

# ==============================
# 7. CHAT
# ==============================
st.title("🤖 Central Test AI Assistant")

query = st.text_input("Асуултаа бичнэ үү:")

if query:
    with st.spinner("🤖 AI бодож байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )

            results = vectorstore.similarity_search(enhance_query(query), k=5)

            if not results:
                st.warning("⚠️ Мэдээлэл олдсонгүй")
            else:
                context = "\n\n".join([
                    clean_text(doc.page_content[:1000])
                    for doc in results
                ])

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.2
                )

                prompt = f"""
Та бол Central Test AI Assistant.

Зөвхөн өгөгдсөн мэдээлэлд үндэслэн хариул.

Хэрэв мэдээлэл байхгүй бол:
"Мэдээлэл хангалтгүй байна"

--- МЭДЭЭЛЭЛ ---
{context}

--- АСУУЛТ ---
{query}
"""

                response = llm.invoke(prompt)

                st.markdown("### 🤖 Хариулт:")
                st.write(response.content)

        except Exception as e:
            st.error(f"❌ Алдаа: {str(e)}")
