import os
import re
import time
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

ARTICLE_PATTERN = re.compile(r"(?:^|\n|\.\s)\s*ماده\s*[\.:]?\s*([۰-۹0-9]{1,3})\b")
MAX_VALID_ARTICLE_NO = 250

TOKENS_PER_CHUNK_MIN = 300
TOKENS_PER_CHUNK_MAX = 500


def preprocess_image_for_ocr(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=15)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    return Image.fromarray(binary)


def extract_article_number(match_text):
    """
    Extracts and validates the article number from the matched text.
    Returns None if the number is implausible (e.g. due to an OCR error),
    preventing corrupted data from entering the database.
    """
    raw_number = match_text.translate(DIGIT_MAP)
    try:
        number = int(raw_number)
    except ValueError:
        return None
    if number <= 0 or number > MAX_VALID_ARTICLE_NO:
        return None
    return number


def split_by_article(full_text):
    """
    Splits text into segments based on the word "Madde" (Article).
    Only treats "Madde" as the start of a new article when it appears
    at the beginning of a line or right after a period/newline -- not
    when it appears as an in-text reference (e.g. "per Article 33").
    This prevents content from one article being wrongly attributed
    to another article referenced within the same sentence.
    """
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    segments = []

    if not matches:
        segments.append({"article_no": None, "text": full_text})
        return segments

    if matches[0].start() > 0:
        preamble = full_text[:matches[0].start()].strip()
        if len(preamble) > 20:
            segments.append({"article_no": None, "text": preamble})

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        segment_text = full_text[start:end].strip()
        article_no = extract_article_number(match.group(1))
        segments.append({"article_no": article_no, "text": segment_text})

    return segments


def token_count_estimate(text):
    return len(text.split())


def chunk_segment_by_tokens(segment_text, max_tokens=TOKENS_PER_CHUNK_MAX):
    """
    Splits a segment (article) into sub-chunks of 300-500 tokens if it is
    too long, as required by the project specification.
    """
    words = segment_text.split()
    if len(words) <= max_tokens:
        return [segment_text]

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_tokens]
        chunks.append(" ".join(chunk_words))
        i += max_tokens
    return chunks


def format_elapsed(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def extract_and_vectorize():
    knowledge_chunks = []
    pipeline_start = time.time()

    pdf_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")]
    total_files = len(pdf_files)

    print("=" * 70)
    print(f"[START] Found {total_files} PDF file(s) to process.")
    print("=" * 70)

    for file_idx, pdf_file in enumerate(pdf_files, start=1):
        file_start = time.time()
        full_path = os.path.join(CORPUS_DIR, pdf_file)

        print(f"\n[FILE {file_idx}/{total_files}] Reading: {pdf_file}")

        print(f"[FILE {file_idx}/{total_files}] Converting PDF pages to images (this may take a while)...")
        pages_images = convert_from_path(
            full_path, dpi=350, poppler_path=r"C:\poppler-26.02.0\Library\bin"
        )
        total_pages = len(pages_images)
        print(f"[FILE {file_idx}/{total_files}] Done. {total_pages} page(s) found.")

        file_chunks_before = len(knowledge_chunks)

        for page_idx, img in enumerate(pages_images, start=1):
            page_start = time.time()

            processed_img = preprocess_image_for_ocr(img)

            raw_text = pytesseract.image_to_string(
                processed_img, lang="fas", config=TESSERACT_CONFIG
            )

            if not raw_text.strip():
                page_elapsed = time.time() - page_start
                print(f"  [PAGE {page_idx}/{total_pages}] Empty page, skipped. ({page_elapsed:.1f}s)")
                continue

            normalized_text = text_cleaner.normalize(raw_text)
            article_segments = split_by_article(normalized_text)

            page_chunk_count = 0
            for seg in article_segments:
                piece = seg["text"]
                if token_count_estimate(piece) < 8:
                    continue

                sub_chunks = chunk_segment_by_tokens(piece)

                for sub_piece in sub_chunks:
                    knowledge_chunks.append({
                        "content": sub_piece,
                        "meta": {
                            "document": pdf_file,
                            "page": page_idx,
                            "article_no": seg["article_no"] if seg["article_no"] is not None else -1,
                        }
                    })
                    page_chunk_count += 1

            page_elapsed = time.time() - page_start
            progress_pct = (page_idx / total_pages) * 100
            print(
                f"  [PAGE {page_idx}/{total_pages}] ({progress_pct:.0f}%) "
                f"OCR done, {page_chunk_count} chunk(s) created. ({page_elapsed:.1f}s)"
            )

        file_elapsed = time.time() - file_start
        file_chunk_total = len(knowledge_chunks) - file_chunks_before
        overall_pct = (file_idx / total_files) * 100
        print(
            f"[FILE {file_idx}/{total_files}] ({overall_pct:.0f}% of files) Finished. "
            f"{file_chunk_total} chunk(s) from this file. Took {format_elapsed(file_elapsed)}."
        )

    if not knowledge_chunks:
        print("\n[ERROR] Text extraction failed. No chunks were created.")
        return

    print("\n" + "=" * 70)
    print(f"[OCR COMPLETE] Total chunks created: {len(knowledge_chunks)}")
    print(f"[OCR COMPLETE] Total time so far: {format_elapsed(time.time() - pipeline_start)}")
    print("=" * 70)

    print("\n[EMBEDDING] Loading embedding model...")
    embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("[EMBEDDING] Model loaded.")

    content_list = [item["content"] for item in knowledge_chunks]
    meta_list = [item["meta"] for item in knowledge_chunks]

    print(f"\n[VECTOR DB] Creating embeddings and storing {len(content_list)} chunks in ChromaDB...")
    print("[VECTOR DB] This step also may take a while depending on chunk count...")

    db_start = time.time()
    Chroma.from_texts(
        texts=content_list,
        embedding=embed_model,
        metadatas=meta_list,
        persist_directory=VECTOR_INDEX_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    db_elapsed = time.time() - db_start

    total_elapsed = time.time() - pipeline_start
    print(f"[VECTOR DB] Done. Took {format_elapsed(db_elapsed)}.")
    print("\n" + "=" * 70)
    print(f"[DONE] Database built successfully!")
    print(f"[DONE] Total pipeline time: {format_elapsed(total_elapsed)}")
    print("=" * 70)


if __name__ == "__main__":
    extract_and_vectorize()