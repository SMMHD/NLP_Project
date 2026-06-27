import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from hazm import Normalizer
from settings import VECTOR_INDEX_DIR, EMBEDDING_MODEL_NAME, GROQ_API_TOKEN

# اعمال تنظیمات صفحه و راست‌چین کردن
st.set_page_config(page_title="سیستم پرسش و پاسخ", page_icon="📚", layout="centered")
st.markdown("""
<style>
    body, .stApp { direction: rtl; text-align: right; font-family: Tahoma, sans-serif; }
    .stTextInput > div > div > div > div > p { text-align: right !important; direction: rtl; }
    .stAlert > div { direction: rtl; text-align: right; }
    .stButton > button { float: right; }
</style>
""", unsafe_allow_html=True)

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

user_input = st.text_input("جستجو:")

if st.button("پیدا کن") and user_input:
    with st.spinner("لطفا صبر کنید..."):
        
        # نرمال‌سازی و اضافه کردن query: مخصوص مدل E5
        clean_input = query_normalizer.normalize(user_input)
        
        # جستجوی عمیق‌تر با k=8 (افزایش شعاع جستجو)
        retrieved_docs = db_instance.similarity_search_with_score(clean_input, k=8)
        
        if not retrieved_docs:
            st.error("پاسخ این سؤال در اسناد موجود یافت نشد.")
        else:
            extracted_context = ""
            references = []
            
            for document, distance_val in retrieved_docs:
                # در فضای cosine مدل E5، فاصله‌های زیر 0.35 معمولاً مرتبط هستند
                # اگر فاصله خیلی زیاد بود (متن بی‌ربط)، آن را نادیده بگیر
                if distance_val < 0.4:
                    extracted_context += f"\nمتن مرجع: {document.page_content}\n"
                    references.append({
                        "doc": document.metadata.get('document', 'نامعلوم'),
                        "pg": document.metadata.get('page', 'نامعلوم'),
                        "score": round(distance_val, 3)
                    })
            
            # اگر بعد از فیلتر هیچ متنی نماند، جواب منفی بده
            if not extracted_context.strip():
                st.error("پاسخ این سؤال در اسناد موجود یافت نشد.")
                st.stop()
                
            ai_prompt = """
            شما یک دستیار هوشمند دانشگاهی هستید. 
            وظیفه شما این است که با دقت متون زیر را بخوانید و ارتباط معنایی آن‌ها را با پرسش کاربر پیدا کنید.
            سپس یک پاسخ کامل، روان و بر اساس متن‌ها ارائه دهید.
            اگر پاسخ در متن‌ها وجود دارد اما کلمات آن کمی متفاوت است، مفهوم را درک کرده و پاسخ دهید.
            فقط اگر هیچ پاسخی (حتی مفهومی) در متون پیدا نکردید بنویسید: "پاسخ این سؤال در اسناد موجود یافت نشد."
            
            متون استخراج شده:
            {context}
            
            پرسش کاربر: {query}
            """
            
            template = ChatPromptTemplate.from_template(ai_prompt)
            pipeline = template | language_model
            
            final_answer = pipeline.invoke({"context": extracted_context, "query": clean_input})
            
            st.success("پاسخ:")
            st.write(final_answer.content)
            
            st.markdown("---")
            st.markdown("**منابع یافت شده:**")
            for ref in references:
                st.caption(f"📄 سند: {ref['doc']} | صفحه: {ref['pg']} (امتیاز: {ref['score']})")