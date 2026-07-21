import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from settings import VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME

embeds = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=VECTOR_INDEX_DIR, embedding_function=embeds)

query = sys.argv[1] if len(sys.argv) > 1 else "حداقل نمره قبولی هر درس در کارشناسی ارشد"

results = db.max_marginal_relevance_search(query, k=8, fetch_k=20)

for i, doc in enumerate(results):
    print("=" * 60)
    print(f"چانک {i+1} | سند: {doc.metadata.get('document')} | صفحه: {doc.metadata.get('page')}")
    print("-" * 60)
    print(doc.page_content)
    print()
