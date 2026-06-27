import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from hazm import Normalizer
from settings import VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME, GROQ_API_TOKEN

# ----------------- تنظیمات صفحه و استایل‌های اختصاصی (UI) -----------------
st.set_page_config(page_title="سیستم پرسش و پاسخ", page_icon="📚", layout="centered")

st.markdown("""
<style>
    /* تنظیمات پایه برای راست‌چین کردن کامل و رفع مشکل خوانایی */
    .stApp {
        direction: rtl;
        text-align: right;
        background-color: #f0f2f6;
    }
    
    /* تنظیم رنگ متن عمومی که سفید نشود */
    p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
        font-family: 'Tahoma', 'Vazir', 'B Nazanin', sans-serif !important;
    }

    /* استایل اختصاصی برای عنوان اصلی */
    h1 {
        color: #0f172a !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        text-align: center !important;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
        margin-bottom: 25px;
    }

    /* استایل ورودی متن (input) */
    .stTextInput > div > div > input {
        border-radius: 6px !important;
        border: 2px solid #cbd5e1 !important;
        padding: 10px 15px !important;
        font-size: 16px !important;
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* استایل دکمه جستجو */
    .stButton > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 5px 25px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        float: right;
        margin-top: 10px;
    }
    .stButton > button:hover {
        background-color: #0f172a !important;
    }

    /* باکس پاسخ سیستم (Success Box) */
    .stAlert {
        background-color: #e0f2fe !important;
        border-right: 5px solid #0284c7 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stAlert p {
        color: #0369a1 !important;
        font-size: 16px !important;
        line-height: 1.8 !important;
    }

    /* باکس‌های رفرنس در پایین صفحه */
    .ref-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-right: 4px solid #64748b;
        padding: 12px 15px;
        margin-bottom: 10px;
        border-radius: 6px;
        font-size: 14px;
        color: #334155;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
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

st.title("📚 سامانه قوانین آموزشی دانشگاه")
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
        لطفاً پاسخ خود را فقط به یک خط محدود نکنید؛ اگر در متن شرایط، تبصره‌ها یا مراحل مختلفی برای سوال کاربر وجود دارد، همه آن‌ها را به صورت دسته‌بندی‌شده و کامل توضیح دهید.
        اگر پاسخ در متن‌ها وجود دارد اما کلمات آن کمی متفاوت است، مفهوم را درک کرده و پاسخ دهید.
        فقط و فقط اگر هیچ پاسخی (حتی مفهومی) در متون پیدا نکردید، عیناً بنویسید: "پاسخ این سؤال در اسناد موجود یافت نشد."
        
        متون استخراج شده:
        {context}
        
        پرسش کاربر: {query}
        """
        
        template = ChatPromptTemplate.from_template(ai_prompt)
        pipeline = template | language_model
        
        # ارسال پرسش و متون به مدل زبانی
        final_answer = pipeline.invoke({"context": extracted_context, "query": clean_input})
        
        st.success("نتیجه بررسی:")
        st.write(final_answer.content)
        
        # نمایش منابع با استایل جدید
        st.markdown("<hr style='border:1px solid #cbd5e1; margin-top:30px;'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#1e293b;'>📑 مستندات و منابع ارجاعی:</h4>", unsafe_allow_html=True)
        
        for idx, ref in enumerate(references):
            st.markdown(
                f"<div class='ref-box'>"
                f"<b>منبع {idx+1}:</b> سند <i>{ref['doc']}</i> | <b>صفحه:</b> {ref['pg']}"
                f"</div>", 
                unsafe_allow_html=True
            )