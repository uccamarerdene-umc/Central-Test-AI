import streamlit as st
import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ✅ FIX
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# ==============================
# 1. ENV LOAD
# ==============================
load_dotenv()

st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

# ==============================
# 2. API KEYS
# ==============================
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

index_name = "centralai-embed-3-small"
dimension = 1536

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API түлхүүр дутуу байна!")
    st.stop()

# ==============================
# 3. LOAD MODELS
# ==============================
@st.cache_resource
def load_models():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

    pc = Pinecone(api_key=pinecone_api_key)

    # Index check
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

    return embeddings, pc

embeddings, pc = load_models()

# ==============================
# 4. SIDEBAR - SYNC
# ==============================
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Sync Data to Pinecone"):
        if not os.path.exists("data"):
            st.error("❌ data хавтас олдсонгүй")
        else:
            with st.spinner("📤 Sync хийж байна..."):
                try:
                    loader = DirectoryLoader(
                        "data",
                        glob="**/*.docx",
                        loader_cls=Docx2txtLoader
                    )
                    docs = loader.load()

                    if not docs:
                        st.warning("⚠️ docx файл олдсонгүй")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=800,
                            chunk_overlap=150
                        )
                        texts = splitter.split_documents(docs)

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
# 5. CHAT UI
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

            results = vectorstore.similarity_search(query, k=5)

            if not results:
                st.warning("⚠️ Мэдээлэл олдсонгүй")
            else:
                context = "\n\n".join([doc.page_content for doc in results])

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.2
                )

                prompt = f"""
Та бол Central Test AI Assistant.

Доорх мэдээлэлд үндэслэн хариул.
Хэрэв байхгүй бол: "Мэдээлэл хангалтгүй байна"

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
