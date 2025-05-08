# 🤖 Multi-PDF Chat App (Local AI Assistant)

A lightweight, privacy-first local chat app that lets you upload multiple PDFs and chat with them — powered by local AI models for summarization and question answering. Works fully offline and designed for low-end machines (8GB RAM friendly).

---

## 🚀 Features

- 📚 **Multi-PDF Upload**: Upload and query multiple PDFs at once.
- 🧠 **Local LLM Support**: Uses lightweight models for summarization and QA.
- 💬 **Chat Interface**: Clean conversational UI built with Streamlit.
- 🔍 **Search Chat History** *(optional)*.
- 🌐 **Translate Chat to Other Languages** *(optional)*.
- 📊 **Chat Stats**: Total messages, word count, etc.
- 🔁 **Undo Last Message**.
- 💾 **Save & Load Sessions** (JSON-based).
- 🧠 **Recall User Inputs**: Assistant can show all your past questions.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Langchain / local LLM)
- **File Handling**: PyMuPDF / PDFMiner
- **Lightweight RAG Setup**: FAISS or in-memory index

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/your-username/multi-pdf-chat-app
cd multi-pdf-chat-app

2. **Create Virtual Environment**
python -m venv venv
venv\Scripts\activate  # On Windows

3. **Install Dependencies**
pip install -r requirements.txt

4. **Run the App**
streamlit run chatapp.py

