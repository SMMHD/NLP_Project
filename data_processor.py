import os
import re
import pytesseract
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from hazm import Normalizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from settings import CORPUS_DIR, VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text_cleaner = Normalizer()

TESSERACT_CONFIG = "--oem 1 --psm 6"

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ASCII_DIGITS = "0123456789"
DIGIT_MAP = str.maketrans(PERSIAN_DIGITS, ASCII_DIGITS)

ARTICLE_PATTERN = re.compile(r"ماده\s*[\.:]?\s*([۰-۹0-9]{1,3})")


def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return Image.fromarray(binary)


def split_by_article(full_text):
    """
    متن هر صفحه را بر اساس کلمه «ماده» به قطعات جداگانه تقسیم می‌کند و
    شماره ماده را با regex (نه با مدل زبانی) استخراج می‌کند تا خطای
    بازتولید عدد توسط LLM به‌طور کامل حذف شود.
    """
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    segments = []

    if not matches:
        segments.append({"article_no": None, "text": full_text})
        return segments

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        segment_text = full_text[start:end].strip()

        raw_number = match.group(1).translate(DIGIT_MAP)
        try:
            article_no = int(raw_number)
        except ValueError:
            article_no = None

        segments.append({"article_no": article_no, "text": segment_text})

    return segments


def extract_and_vectorize():
    knowledge_chunks = []
    print("شروع پردازش فایل‌های PDF (OCR)...")

    for pdf_file in os.listdir(CORPUS_DIR):
        if pdf_file.endswith(".pdf"):
            full_path = os.path.join(CORPUS_DIR, pdf_file)
            print(f"در حال خواندن فایل: {pdf_file}")

            pages_images = convert_from_path(
                full_path, dpi=350, poppler_path=r"C:\poppler-26.02.0\Library\bin"
            )

            for index, img in enumerate(pages_images):
                processed_img = preprocess_image_for_ocr(img)

                raw_text = pytesseract.image_to_string(
                    processed_img, lang="fas", config=TESSERACT_CONFIG
                )

                if not raw_text.strip():
                    continue

                normalized_text = text_cleaner.normalize(raw_text)

                article_segments = split_by_article(normalized_text)

                for seg in article_segments:
                    piece = seg["text"]
                    if len(piece) < 20:
                        continue

                    if len(piece) > 1800:
                        sub_chunks = [piece[j:j + 1500] for j in range(0, len(piece), 1500)]
                    else:
                        sub_chunks = [piece]

                    for sub_piece in sub_chunks:
                        knowledge_chunks.append({
                            "content": sub_piece,
                            "meta": {
                                "document": pdf_file,
                                "page": index + 1,
                                "article_no": seg["article_no"] if seg["article_no"] is not None else -1,
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