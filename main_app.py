import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import DB_DIR, EMBEDDING_MODEL, GROQ_API_KEY

# بارگذاری مدل‌ها و دیتابیس
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
# استفاده از allow_dangerous_deserialization برای لود کردن FAISS [web:57]
knowledge_db = FAISS.load_local(DB_DIR, embed_model, allow_dangerous_deserialization=True)
llm_engine = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

# llm_engine = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=GROQ_API_KEY)
# llm_engine = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", groq_api_key=GROQ_API_KEY)

def generate_answer(user_question, history):
    # جستجو در دیتابیس FAISS (تغییر k به 4 برای ایجاد تفاوت)
    # search_results = knowledge_db.similarity_search_with_score(f"query: {user_question}", k=6)
    search_results = knowledge_db.similarity_search_with_score(user_question, k=6)
    
    if not search_results:
        return "پاسخ این سؤال در اسناد موجود یافت نشد."
        
    context_text = ""
    sources_info = "\n\n📚 منابع استخراج شده:\n"
    
    for doc, distance in search_results:
    # فقط متونی را نگه دار که فاصله شباهت منطقی دارند
        context_text += f"\nمتن: {doc.page_content}\n"
        sources_info += f"- سند: {doc.metadata.get('doc_name')} | صفحه: {doc.metadata.get('page_number')} | امتیاز شباهت: {round(distance, 4)}\n"
            
    # اگر بعد از فیلتر کردن هیچ متنی باقی نماند:
    if not context_text.strip():
        return "پاسخ این سؤال در اسناد موجود یافت نشد."

    # تغییر لحن پرامپت
    sys_prompt = """
    شما سیستم هوش مصنوعی و راهنمای آموزشی دانشگاه شهید باهنر هستید.
    با دقت متون زیر را تحلیل کنید و بر اساس آن‌ها یک پاسخ روان و کامل به پرسش دانشجو بدهید.
    اگر پاسخ در متن‌ها وجود دارد اما کلمات آن متفاوت است، مفهوم اصلی را درک کرده و پاسخ دهید.
    فقط و فقط اگر هیچ ارتباطی بین متون استخراج شده و پرسش دانشجو پیدا نکردید، عیناً این جمله را بنویسید: "پاسخ این سؤال در اسناد موجود یافت نشد."
    هیچ اطلاعاتی خارج از متون زیر به پاسخ اضافه نکنید.
    
    متون استخراج شده:
    {context}
    
    پرسش دانشجو: {query}
    """
    
    prompt_builder = ChatPromptTemplate.from_template(sys_prompt)
    chain = prompt_builder | llm_engine
    
    bot_response = chain.invoke({"context": context_text, "query": user_question})
    
    # ترکیب پاسخ مدل و منابع برای نمایش در رابط کاربری
    final_output = bot_response.content + sources_info
    return final_output

# ساخت رابط کاربری با Gradio
custom_theme = gr.themes.Soft()

# ساخت رابط کاربری با Gradio (نسخه سازگار با تمامی ورژن‌ها)
ui = gr.ChatInterface(
    fn=generate_answer,
    title="🤖 راهنمای هوشمند آیین‌نامه‌های دانشگاه باهنر",
    description="سوالات خود را در مورد قوانین و مقررات آموزشی بپرسید."
)

if __name__ == "__main__":
    ui.launch(inbrowser=True)