import streamlit as st
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# Load local `.env` for dev (Streamlit secrets will still take priority).
load_dotenv()

# ==============================
# 1. APP CONFIG
# ==============================
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

# Prefer Streamlit secrets; fallback to environment variables (for local runs).
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# NOTE: `text-embedding-3-small` dimension is 1536.
index_name = "centralai-embed-3-small"
dimension = 1536  # embedding хэмжээ

# ==============================
# 2. CHECK API KEYS
# ==============================
# Google (Gemini) + OpenAI embeddings + Pinecone шаардлагатай.
if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API түлхүүрүүд тохируулагдаагүй байна! (GOOGLE_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY шалгана уу)")
    st.stop()

# ==============================
# 3. LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    # OpenAI embeddings: `text-embedding-3-small`
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    pc = Pinecone(api_key=pinecone_api_key)

    # Index байгаа эсэх шалгах
    if index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return embeddings, pc

embeddings, pc = load_models()

# ==============================
# 4. SIDEBAR - SYNC DATA
# ==============================
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Sync Data to Pinecone"):
        if not os.path.exists("Data"):
            st.error("❌ 'Data' хавтас олдсонгүй!")
        else:
            with st.spinner("📤 Pinecone-д өгөгдөл илгээж байна..."):
                try:
                    loader = DirectoryLoader(
                        "Data",
                        glob="**/*.docx",
                        loader_cls=Docx2txtLoader
                    )
                    docs = loader.load()

                    if not docs:
                        st.warning("⚠️ .docx файл олдсонгүй.")
                    else:
                        # Chunking
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=800,
                            chunk_overlap=150
                        )
                        texts = splitter.split_documents(docs)

                        # Vector store
                        vectorstore = PineconeVectorStore.from_documents(
                            documents=texts,
                            embedding=embeddings,
                            index_name=index_name,
                            pinecone_api_key=pinecone_api_key
                        )

                        st.success(f"✅ {len(texts)} chunk амжилттай хадгалагдлаа!")

                except Exception as e:
                    st.error(f"❌ Sync failed: {str(e)}")

# ==============================
# 5. CHAT UI
# ==============================
st.title("🤖 Central Test AI Assistant")

query = st.text_input("Асуултаа бичнэ үү:")

if query:
    with st.spinner("🤖 AI бодож байна..."):
        try:
            # Vector store load
            vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )

            # Search
            results = vectorstore.similarity_search(query, k=5)

            if not results:
                st.warning("⚠️ Холбогдох мэдээлэл олдсонгүй.")
                st.stop()

            context = "\n\n".join([doc.page_content for doc in results])

            # LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.2
            )

            prompt = f"""
Та бол Central Test AI Assistant.

Доорх мэдээлэлд үндэслэн асуултад хариул.
Хэрэв мэдээлэл дутуу бол "Мэдээлэл хангалтгүй байна" гэж хэл.

--- МЭДЭЭЛЭЛ ---
{context}

--- АСУУЛТ ---
{query}
"""

            response = llm.invoke(prompt)

            st.markdown("### 🤖 Хариулт:")
            st.write(response.content)

        except Exception as e:
            st.error(f"❌ Алдаа гарлаа: {str(e)}")
