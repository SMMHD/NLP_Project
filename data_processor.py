import os
import pytesseract
from pdf2image import convert_from_path
from hazm import Normalizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from settings import CORPUS_DIR, VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME

# تنظیم مسیر Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text_cleaner = Normalizer()

def extract_and_vectorize():
    knowledge_chunks = []
    print("شروع پردازش فایل‌های PDF (OCR)...")

    for pdf_file in os.listdir(CORPUS_DIR):
        if pdf_file.endswith(".pdf"):
            full_path = os.path.join(CORPUS_DIR, pdf_file)
            print(f"در حال خواندن فایل: {pdf_file}")
            
            # تبدیل به تصویر
            pages_images = convert_from_path(full_path, dpi=200, poppler_path=r"C:\poppler-26.02.0\Library\bin")
            
            for index, img in enumerate(pages_images):
                raw_text = pytesseract.image_to_string(img, lang='fas')
                
                if not raw_text.strip():
                    continue
                
                normalized_text = text_cleaner.normalize(raw_text)
                
                # قطعه‌بندی با ابعاد متفاوت
                chunker = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " "]
                )
                
                split_results = chunker.split_text(normalized_text)
                
                for piece in split_results:
                    knowledge_chunks.append({
                        "content": piece,
                        "meta": {
                            "document": pdf_file,
                            "page": index + 1
                        }
                    })

    if not knowledge_chunks:
        print("خطا: استخراج متن ناموفق بود.")
        return

    print("ایجاد و ذخیره‌سازی دیتابیس ChromaDB...")
    embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    content_list = [item["content"] for item in knowledge_chunks]
    meta_list = [item["meta"] for item in knowledge_chunks]
    
    Chroma.from_texts(
        texts=content_list,
        embedding=embed_model,
        metadatas=meta_list,
        persist_directory=VECTOR_INDEX_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("دیتابیس ساخته شد!")

if __name__ == "__main__":
    extract_and_vectorize()