import streamlit as st
import base64
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from hazm import Normalizer
import re
from settings import VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME, GROQ_API_TOKEN

st.set_page_config(page_title="سیستم پرسش و پاسخ", page_icon="📚", layout="centered")


def load_font_base64(font_path):
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


BNAZANIN_FONT_PATH = "fonts/BNazanin.ttf"
bnazanin_b64 = load_font_base64(BNAZANIN_FONT_PATH)

if bnazanin_b64:
    font_face_rule = "@font-face { font-family: 'BNazaninCustom'; src: url(data:font/ttf;base64," + bnazanin_b64 + ") format('truetype'); unicode-range: U+0600-06FF, U+FB8A, U+067E, U+0686, U+06AF, U+200C, U+200F; font-weight: normal; font-style: normal; }"
else:
    font_face_rule = "@font-face { font-family: 'BNazaninCustom'; src: local('B Nazanin'); unicode-range: U+0600-06FF, U+FB8A, U+067E, U+0686, U+06AF, U+200C, U+200F; font-weight: normal; font-style: normal; }"

css = """
<meta name="color-scheme" content="light">
<style>
""" + font_face_rule + """
@font-face { font-family: 'BNazaninCustom'; src: local('Times New Roman'); unicode-range: U+0000-007F, U+0030-0039; font-weight: normal; font-style: normal; }
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
[data-testid="stDecoration"] {visibility: hidden;}
[data-testid="stStatusWidget"] {visibility: hidden;}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"], section.main, .main .block-container {
background-color: #ffffff !important;
color: #262730 !important;
color-scheme: light !important;
}
html, body, [class*="css"], .stApp {
direction: rtl;
font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
}
.stMarkdown, .stAlert, .stSuccess, .stError, .stSpinner, p, h1, h2, h3, h4, h5, h6, span, div, li, label {
direction: rtl;
text-align: right;
font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
color: #262730 !important;
}
.app-title {
text-align: center;
font-size: 2.1rem;
font-weight: bold;
margin-top: 0.5rem;
margin-bottom: 1.8rem;
color: #262730 !important;
}
[data-testid="stTextInput"] label {
text-align: center !important;
width: 100%;
display: block;
direction: rtl;
}
.stTextInput > div > div > input {
direction: rtl;
text-align: right;
background-color: #ffffff !important;
color: #262730 !important;
font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
}
div[data-testid="stButton"] {
margin-top: 0.8rem;
}
.stButton > button {
direction: rtl;
font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
font-weight: bold !important;
font-size: 1.05rem;
padding: 0.5rem 2.2rem;
width: 100%;
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)


def clean_non_persian_chars(text):
    """
    حذف کاراکترهای ناخواسته چینی/ژاپنی/کره‌ای (CJK) که گاهی مدل‌های زبانی
    به‌اشتباه در متن فارسی تولید می‌کنند. حروف فارسی، عربی، انگلیسی، اعداد
    و علائم نگارشی رایج حفظ می‌شوند.
    """
    cjk_pattern = re.compile(
        r'[\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]'
    )
    cleaned = cjk_pattern.sub('', text)
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    return cleaned


query_normalizer = Normalizer()


@st.cache_resource
def init_system():
    embeds = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    chroma_db = Chroma(persist_directory=VECTOR_INDEX_DIR, embedding_function=embeds)
    llm_model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_TOKEN)
    return chroma_db, llm_model


db_instance, language_model = init_system()

st.markdown('<div class="app-title">سامانه قوانین آموزشی دانشگاه</div>', unsafe_allow_html=True)

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    user_input = st.text_input("جستجو در آیین‌نامه‌ها:")
    search_clicked = st.button("جستجو", use_container_width=True)

if search_clicked and user_input:
    with st.spinner("لطفا صبر کنید... سیستم در حال تحلیل مستندات است..."):

        clean_input = query_normalizer.normalize(user_input)

        retrieved_docs_mmr = db_instance.max_marginal_relevance_search(clean_input, k=8, fetch_k=20)

        if not retrieved_docs_mmr:
            st.error("پاسخ این سؤال در اسناد موجود یافت نشد.")
            st.stop()

        extracted_context = ""
        references = []
        for document in retrieved_docs_mmr:
            extracted_context += "\nمتن مرجع: " + document.page_content + "\n"
            references.append({
                "doc": document.metadata.get('document', 'نامعلوم'),
                "pg": document.metadata.get('page', 'نامعلوم'),
                "article": document.metadata.get('article_no', -1),
            })

        ai_prompt = """
        شما یک دستیار هوشمند، صبور و دقیق دانشگاهی هستید که فقط بر اساس متن‌های زیر پاسخ می‌دهید.

        قوانین بسیار مهم و اجباری:
        1. هر عدد، شماره ماده، شماره تبصره یا شماره بند را دقیقاً و کلمه‌به‌کلمه همان‌طور که در متن مرجع نوشته شده کپی کنید. هرگز شماره‌ها را از حافظه خود بازسازی یا حدس نزنید.
        2. قبل از نوشتن هر شماره ماده در پاسخ، آن را با متن مرجع مطابقت دهید تا مطمئن شوید دقیقاً همان عدد است.
        3. اگر عین متن مرجع را نقل می‌کنید (مثلاً شماره ماده)، آن را داخل گیومه یا به‌صورت مشخص از توضیح خودتان جدا کنید.
        4. یک پاسخ کامل، جامع و دارای جزئیات کافی بر اساس متن‌ها ارائه دهید و فقط به یک خط محدود نکنید.
        5. اگر در متن شرایط، تبصره‌ها یا مراحل مختلفی برای سوال کاربر وجود دارد، همه آن‌ها را به صورت دسته‌بندی‌شده و کامل توضیح دهید.
        6. اگر پاسخ در متن‌ها وجود دارد اما کلمات آن کمی متفاوت است، مفهوم را درک کرده و پاسخ دهید، اما شماره‌های ماده را دقیقاً همان‌طور که نوشته شده حفظ کنید.
        7. فقط و فقط اگر هیچ پاسخی (حتی مفهومی) در متون پیدا نکردید، عیناً بنویسید:
        "پاسخ این سؤال در اسناد موجود یافت نشد."

        متون استخراج شده:
        {context}

        پرسش کاربر:
        {query}
        """

        template = ChatPromptTemplate.from_template(ai_prompt)
        pipeline = template | language_model

        final_answer = pipeline.invoke({"context": extracted_context, "query": clean_input})

        st.success("نتیجه بررسی:")
        cleaned_answer = clean_non_persian_chars(final_answer.content)
        st.write(cleaned_answer)

        if references:
            st.markdown("### 📄 منابع استفاده‌شده:")
            for ref in references:
                article_display = f" | ماده: **{ref['article']}**" if ref.get('article', -1) != -1 else ""
                st.markdown("- سند: **" + str(ref['doc']) + "** | صفحه: **" + str(ref['pg']) + "**" + article_display)