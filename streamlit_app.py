import streamlit as st
import os
import unicodedata

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔥 MULTI FILE LOADER
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    DirectoryLoader
)

# ✅ STABLE SPLITTER (NO SPACY)
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
# 2. TEXT CLEAN (🔥 MONGOLIAN SAFE)
# ==============================
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    return text.encode("utf-8", "ignore").decode("utf-8")

# ==============================
# 3. LOAD ALL FILE TYPES
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
            loader = DirectoryLoader(
                path,
                glob=pattern,
                loader_cls=loader_cls,
                show_progress=True
            )
            docs.extend(loader.load())

    return docs

# ==============================
# 4. LOAD MODELS
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
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return embeddings

embeddings = load_models()

# ==============================
# 5. SIDEBAR SYNC
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
                        # 🔥 Монгол optimized chunking
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=120,
                            separators=["\n\n", "\n", ".", "!", "?", " "]
                        )

                        texts = splitter.split_documents(docs)

                        # 🔥 CLEAN TEXT
                        for doc in texts:
                            doc.page_content = clean_text(doc.page_content)

                        PineconeVectorStore.from_documents(
                            documents=texts,
                            embedding=embeddings,
                            index_name=index_name,
                            pinecone_api_key=pinecone_api_key
                        )

                        st.success(f"✅ {len(texts)} chunk хадгалагдлаа")

                except Exception as e:
                    st.error(f"❌ Sync алдаа: {str(e)}")

# ==============================
# 6. QUERY OPTIMIZATION
# ==============================
def enhance_query(query):
    return f"""
Дараах асуултыг ойлгомжтой утгаар хайлт хийхэд ашиглана:

{query}
"""

# ==============================
# 7. CHAT UI
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

            enhanced_query = enhance_query(query)

            results = vectorstore.similarity_search(enhanced_query, k=5)

            if not results:
                st.warning("⚠️ Мэдээлэл олдсонгүй")
            else:
                context = "\n\n".join([
                    doc.page_content[:1000]
                    for doc in results
                ])

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.2
                )

                prompt = f"""
Та бол Central Test AI Assistant.

Та зөвхөн өгөгдсөн мэдээлэлд үндэслэн Монгол хэл дээр товч, тодорхой хариул.

Хэрэв мэдээлэл байхгүй бол:
"Мэдээлэл хангалтгүй байна" гэж хариул.

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
