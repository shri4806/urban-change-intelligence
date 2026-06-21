from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "rag/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

query = """
25.26% urban change detected.
17 regions detected.
Largest hotspot located in upper-left region.
"""

results = vectorstore.similarity_search(
    query,
    k=3
)

for i, doc in enumerate(results):

    print("\n")
    print("=" * 50)

    print(f"RESULT {i+1}")

    print("=" * 50)

    print(doc.page_content[:1000])