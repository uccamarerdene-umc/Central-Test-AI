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
    UnstructuredPowerPointLoader
)

from langchain.schema import Document as LCDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ==============================
# ENV
# ==============================
load_dotenv()
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

index_name = "centralai-embed-3-small"

if not google_api_key or not pinecone_api_key or not openai_api_key:
    st.error("❌ API түлхүүр дутуу байна!")
    st.stop()

# ==============================
# CLEAN TEXT (Updated for safer encoding)
# ==============================
def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    
    # Монгол үсэг болон тусгай тэмдэгтүүдийг UTF-8 хэвээр үлдээж цэвэрлэх
    text = unicodedata.normalize("NFKC", text)
    # ASCII-д байхгүй урт зураасыг энгийн зураасаар солих
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text

# ==============================
# SAFE DOCX LOADER
# ==============================
def load_docx_safe(file_path):
    try:
        # Docx2txtLoader нь заримдаа дотооддоо ascii ашиглах гээд байдаг тул 
        # try-except дотор текстээ clean хийх нь зөв
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        safe_docs = []
        for d in docs:
            safe_text = clean_text(d.page_content)
            safe_docs.append(
                LCDocument(
                    page_content=safe_text,
                    metadata={"source": file_path}
                )
            )
        return safe_docs
    except Exception as e:
        st.warning(f"⚠️ DOCX алдаа: {file_path} → {e}")
        return []

# ==============================
# LOAD ALL DOCUMENTS
# ==============================
def load_all_documents():
    docs = []
    if not os.path.exists("data"):
        return []

    for root, _, files in os.walk("data"):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".docx"):
                    docs.extend(load_docx_safe(path))
                elif file.endswith(".pdf"):
                    loaded_pdf = PyPDFLoader(path).load()
                    for p in loaded_pdf:
                        p.page_content = clean_text(p.page_content)
                    docs.extend(loaded_pdf)
                elif file.endswith(".txt"):
                    # TXT файлыг заавал UTF-8-оор унших
                    docs.extend(TextLoader(path, encoding="utf-8").load())
                elif file.endswith(".pptx"):
                    loaded_pptx = UnstructuredPowerPointLoader(path).load()
                    for p in loaded_pptx:
                        p.page_content = clean_text(p.page_content)
                    docs.extend(loaded_pptx)
            except Exception as e:
                st.warning(f"⚠️ Алдаа ({file}): {e}")
    return docs

# ==============================
# MODELS
# ==============================
@st.cache_resource
def load_models():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Index шалгах
    indexes = [i["name"] for i in pc.list_indexes()]
    if index_name not in indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return embeddings

embeddings = load_models()

# ==============================
# SIDEBAR SYNC
# ==============================
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🔄 Sync Data to Pinecone"):
        if not os.path.exists("data"):
            st.error("❌ data хавтас олдсонгүй")
        else:
            with st.spinner("📤 Sync хийж байна..."):
                try:
                    all_docs = load_all_documents()
                    
                    if not all_docs:
                        st.warning("⚠️ Хавтсанд тохирох файл алга.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=120
                        )
                        texts = splitter.split_documents(all_docs)

                        # Pinecone-д холбогдох
                        vectorstore = PineconeVectorStore(
                            index_name=index_name,
                            embedding=embeddings,
                            pinecone_api_key=pinecone_api_key
                        )

                        safe_contents = []
                        metadatas = []
                        ids = []

                        for doc in texts:
                            content = clean_text(doc.page_content)
                            source = clean_text(doc.metadata.get("source", "unknown"))
                            
                            safe_contents.append(content)
                            metadatas.append({"source": source})
                            ids.append(str(uuid4()))

                        # Багцаар нь нэмэх
                        vectorstore.add_texts(
                            texts=safe_contents,
                            metadatas=metadatas,
                            ids=ids
                        )
                        st.success(f"✅ {len(safe_contents)} хэсэг хадгалагдлаа")

                except Exception as e:
                    # Энд гарч байсан ASCII алдааг clean_text барих ёстой
                    st.error(f"❌ Sync алдаа: {str(e)}")

# ==============================
# CHAT UI
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
                context = "\n\n".join([clean_text(doc.page_content) for doc in results])

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=google_api_key,
                    temperature=0.2
                )

                prompt = f"""Та бол Central Test AI Assistant. Доорх мэдээлэлд үндэслэн хариул:
                {context}
                Асуулт: {query}"""

                response = llm.invoke(prompt)
                st.markdown("### 🤖 Хариулт:")
                st.write(response.content)
        except Exception as e:
            st.error(f"❌ Алдаа: {str(e)}")
