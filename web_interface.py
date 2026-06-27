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
        search_term = f"query: {clean_input}"
        
        # استفاده از جستجوی معمولی (طبق پروژه خودت)
        retrieved_docs = db_instance.similarity_search_with_score(search_term, k=6)
        
        if not retrieved_docs:
            st.error("پاسخ این سؤال در اسناد موجود یافت نشد.")
        else:
            extracted_context = ""
            references = []
            
            for document, distance_val in retrieved_docs:
                extracted_context += f"\nمتن مرجع: {document.page_content}\n"
                references.append({
                    "doc": document.metadata.get('document', 'نامعلوم'),
                    "pg": document.metadata.get('page', 'نامعلوم'),
                    "score": round(distance_val, 3)
                })
                
            ai_prompt = """
            شما یک راهنمای دانشگاهی هستید. فقط بر اساس متون استخراج شده زیر به کاربر پاسخ دهید.
            در صورتی که جواب در این متون نیست، دقیقا بنویسید: "پاسخ این سؤال در اسناد موجود یافت نشد."
            
            متون:
            {context}
            
            پرسش: {query}
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