import streamlit as st
import base64
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from hazm import Normalizer
from settings import VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME, GROQ_API_TOKEN

# ----------------- تنظیمات صفحه و استایل‌های اختصاصی (UI) -----------------
st.set_page_config(page_title="سیستم پرسش و پاسخ", page_icon="\U0001F4DA", layout="centered")


def load_font_base64(font_path: str) -> str:
    """
    فایل فونت را می‌خواند و به رشته base64 تبدیل می‌کند تا بتوان
    آن را مستقیم داخل CSS جای داد (روی هر دستگاهی درست نمایش داده شود).
    """
    if os.path.exists(font_path):
        with open(font_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


# مسیر فایل فونت B Nazanin را اینجا وارد کنید
# (فایل .ttf را در پوشه fonts کنار همین فایل قرار دهید)
BNAZANIN_FONT_PATH = "fonts/BNazanin.ttf"
bnazanin_b64 = load_font_base64(BNAZANIN_FONT_PATH)

if bnazanin_b64:
    font_face_rule = f"""
    @font-face {{
        font-family: 'BNazaninCustom';
        src: url(data:font/ttf;base64,{bnazanin_b64}) format('truetype');
        unicode-range: U+0600-06FF, U+FB8A, U+067E, U+0686, U+06AF, U+200C, U+200F;
        font-weight: normal;
        font-style: normal;
    }}
    """
else:
    # اگر فایل فونت پیدا نشد، از فونت نصب‌شده روی سیستم استفاده می‌شود
    font_face_rule = """
    @font-face {
        font-family: 'BNazaninCustom';
        src: local('B Nazanin');
        unicode-range: U+0600-06FF, U+FB8A, U+067E, U+0686, U+06AF, U+200C, U+200F;
        font-weight: normal;
        font-style: normal;
    }
    """

st.markdown(f"""
<style>
{font_face_rule}

@font-face {{
    font-family: 'BNazaninCustom';
    src: local('Times New Roman');
    unicode-range: U+0000-007F, U+0030-0039;
    font-weight: normal;
    font-style: normal;
}}

html, body, [class*="css"], .stApp {{
    direction: rtl;
    text-align: right;
    font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
}}

.stTextInput > label, .stTextInput > div > div > input {{
    direction: rtl;
    text-align: right;
    font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
}}

.stButton > button {{
    direction: rtl;
    font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
}}

.stMarkdown, .stAlert, .stSuccess, .stError, .stSpinner,
p, h1, h2, h3, h4, h5, h6, span, div, li, label {{
    direction: rtl;
    text-align: right;
    font-family: 'BNazaninCustom', 'Times New Roman', sans-serif !important;
}}

/* اعداد و کاراکترهای انگلیسی داخل متن به‌صورت خودکار با unicode-range مدیریت می‌شوند */
</style>
""", unsafe_allow_html=True)
# -------------------------------------------------------------------------

query_normalizer = Normalizer()


@st.cache_resource
def init_system():
    embeds = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    chroma_db = Chroma(persist_directory=VECTOR_INDEX_DIR, embedding_function=embeds)
    llm_model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_TOKEN)
    return chroma_db, llm_model


db_instance, language_model = init_system()

st.title("\U0001F4DA سامانه قوانین آموزشی دانشگاه")
st.write("لطفا سوال خود را مطرح کنید:")

user_input = st.text_input("جستجو در آیین‌نامه‌ها:")

if st.button("جستجو") and user_input:
    with st.spinner("لطفا صبر کنید... سیستم در حال تحلیل مستندات است..."):

        # ۱. نرمال‌سازی سؤال کاربر (رفع مشکل فاصله‌ها و نیم‌فاصله‌ها)
        clean_input = query_normalizer.normalize(user_input)

        # ۲. استفاده از جستجوی پیشرفته MMR به جای جستجوی ساده
        # این متد 20 متن را بررسی کرده و 8 متن که بیشترین تنوع و ارتباط را دارند انتخاب می‌کند
        retrieved_docs_mmr = db_instance.max_marginal_relevance_search(clean_input, k=8, fetch_k=20)

        if not retrieved_docs_mmr:
            st.error("پاسخ این سؤال در اسناد موجود یافت نشد.")
            st.stop()

        extracted_context = ""
        references = []
        for document in retrieved_docs_mmr:
            extracted_context += f"\nمتن مرجع: {document.page_content}\n"
            references.append({
                "doc": document.metadata.get('document', 'نامعلوم'),
                "pg": document.metadata.get('page', 'نامعلوم'),
                # در حالت MMR فاصله مستقیم محاسبه نمی‌شود، پس فقط رفرنس را ثبت می‌کنیم
            })

        # ۳. پرامپت اختصاصی برای دریافت پاسخ‌های کامل‌تر و تشریحی
        ai_prompt = """
        شما یک دستیار هوشمند، صبور و دقیق دانشگاهی هستید.
        وظیفه شما این است که با دقت متون زیر را بخوانید و ارتباط معنایی آن‌ها را با پرسش کاربر پیدا کنید.
        سپس یک پاسخ کامل، جامع و دارای جزئیات کافی بر اساس متن‌ها ارائه دهید.
        لطفاً پاسخ خود را فقط به یک خط محدود نکنید؛ اگر در متن شرایط، تبصره‌ها یا مراحل مختلفی برای سوال کاربر وجود دارد،
        همه آن‌ها را به صورت دسته‌بندی‌شده و کامل توضیح دهید.
        اگر پاسخ در متن‌ها وجود دارد اما کلمات آن کمی متفاوت است، مفهوم را درک کرده و پاسخ دهید.
        فقط و فقط اگر هیچ پاسخی (حتی مفهومی) در متون پیدا نکردید، عیناً بنویسید:
        "پاسخ این سؤال در اسناد موجود یافت نشد."

        متون استخراج شده:
        {context}

        پرسش کاربر:
        {query}
        """

        template = ChatPromptTemplate.from_template(ai_prompt)
        pipeline = template | language_model

        # ارسال پرسش و متون به مدل زبانی
        final_answer = pipeline.invoke({"context": extracted_context, "query": clean_input})

        st.success("نتیجه بررسی:")
        st.write(final_answer.content)

        # نمایش منابع با استایل جدید
        if references:
            st.markdown("### \U0001F4C4 منابع استفاده‌شده:")
            for ref in references:
                st.markdown(f"- سند: **{ref['doc']}** | صفحه: **{ref['pg']}**")
