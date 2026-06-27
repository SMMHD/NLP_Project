import os
import pytesseract
from pdf2image import convert_from_path
from hazm import Normalizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_DIR, DB_DIR, EMBEDDING_MODEL

# مسیر Tesseract را تنظیم کن
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def build_knowledge_base():
    text_normalizer = Normalizer()
    all_chunks = []
    
    print("شروع خواندن اسناد دانشگاهی...")
    for file_name in os.listdir(DATA_DIR):
        if file_name.endswith(".pdf"):
            full_path = os.path.join(DATA_DIR, file_name)
            print(f"در حال پردازش: {file_name}")
            
            # نکته: مسیر poppler_path را در صورت نیاز مثل پروژه قبلی اضافه کن
            images = convert_from_path(full_path, dpi=200, poppler_path = r"C:\poppler-26.02.0\Library\bin")
            
            for index, img in enumerate(images):
                page_text = pytesseract.image_to_string(img, lang='fas')
                
                if not page_text.strip():
                    continue
                    
                clean_txt = text_normalizer.normalize(page_text)
                
                # تغییر پارامترهای Chunking برای متفاوت بودن نتایج
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", ".", " "]
                )
                split_texts = splitter.split_text(clean_txt)
                
                for chunk in split_texts:
                    all_chunks.append({
                        "text": chunk, 
                        "metadata": {"doc_name": file_name, "page_number": index + 1}
                    })

    if not all_chunks:
        print("هیچ متنی پیدا نشد.")
        return

    print("تولید بردارها و ذخیره در پایگاه داده FAISS...")
    embeddings_tool = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # تغییر دیتابیس از Chroma به FAISS [web:57]
    texts_list = [c["text"] for c in all_chunks]
    meta_list = [c["metadata"] for c in all_chunks]
    
    vector_database = FAISS.from_texts(texts_list, embeddings_tool, metadatas=meta_list)
    vector_database.save_local(DB_DIR)
    
    print("پایگاه داده با موفقیت ایجاد و ذخیره شد.")

if __name__ == "__main__":
    build_knowledge_base()