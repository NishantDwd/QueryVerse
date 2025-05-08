import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from io import BytesIO
from fpdf import FPDF

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_chat_as_txt():
    return "\n\n".join([f"User: {chat['question']}\nAssistant: {chat['answer']}" for chat in st.session_state.chat_history])


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def generate_content_with_gemini(question):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    if "summarize" in user_question.lower():
        docs = new_db.similarity_search("summary")
    else:
        docs = new_db.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are a helpful assistant. Based on the following PDF content, answer the question or generate a summary:

Context:
{context}

Question:
{user_question}
"""

    response_text = generate_content_with_gemini(prompt)

    # 🔥 ADDED: Append question and answer to session state
    st.session_state.chat_history.append({"question": user_question, "answer": response_text})
    return response_text

def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        with st.spinner("Thinking..."):
            user_input(user_question)

    # ✅ Now display all chat history (including the latest one)
    with st.container():
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(chat["answer"])

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        if st.session_state.chat_history:
            st.download_button("📄 Download Chat as TXT", get_chat_as_txt(), file_name="chat_history.txt")

        st.write("---")
        st.image("img/footer.jpg")
        st.write("AI App created by @ QueryVerse")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © Team QueryVerse | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main() 

