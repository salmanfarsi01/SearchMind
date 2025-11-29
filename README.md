DocMind — RAG Document QA Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot for querying your PDFs & DOCX files.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue" /> <img src="https://img.shields.io/badge/Flask-Backend-red" /> <img src="https://img.shields.io/badge/Pinecone-VectorDB-purple" /> <img src="https://img.shields.io/badge/OpenAI/AWS-Embeddings-00aaff" /> <img src="https://img.shields.io/badge/UI-TailwindCSS-0ea5e9" /> </p>
🚀 About the Project

DocMind is a lightweight, fast, and modern RAG-based chatbot that lets users:

Upload PDF or DOCX documents

Extract text + chunk using recursive splitting

Generate embeddings using OpenAI/AWS-compatible embedding models

Store vectors in Pinecone

Ask natural language questions directly from uploaded documents

Enjoy a clean, modern UI with TailwindCSS

Perfect for:
📘 Research papers · 📄 Contracts · 🧠 Knowledge bases · 🔍 Document analysis

<img width="1183" height="673" alt="image" src="https://github.com/user-attachments/assets/cb2a8716-ed17-4570-b263-aae1c7ba195a" />


Replace the images below with your real screenshots

📥 Upload Documents

💬 Chat Interface

✨ Features

📄 Upload & parse PDF / DOCX

🔍 Smart recursive text chunking

🧠 Embeddings via text-embedding-3-small (OpenAI / AWS)

📦 Vector storage using Pinecone (1536 dims)

🤖 RAG-powered chat with relevant document chunks

🎨 Beautiful TailwindCSS user interface

🗂 Organized file/manifest management system

🌍 Instant public access using ngrok

⚙️ Fully configurable using .env

🏗️ Tech Stack
Layer	Technology
Backend	Flask
Embeddings	OpenAI / AWS OpenAI-compatible
Vector DB	Pinecone
Frontend	TailwindCSS + Vanilla JS
File Parsing	PyPDF2, python-docx
Environment	python-dotenv
Deployment	ngrok / Render / Railway

<img width="598" height="284" alt="image" src="https://github.com/user-attachments/assets/02e11504-07b0-472e-b157-1794a14e4076" />


⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/docmind.git
cd docmind

2️⃣ Create a virtual environment
python -m venv venv

3️⃣ Activate venv

Windows:

venv\Scripts\activate


macOS/Linux:

source venv/bin/activate

4️⃣ Install dependencies
pip install -r requirements.txt

🔑 Environment Variables (.env)

Create a .env file in the project root:

# OpenAI or AWS-compatible OpenAI API
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Pinecone
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=rag-chat-index

# Models
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-mini

▶️ Run the Application
python app.py


App runs at:

http://127.0.0.1:5000

🌍 Expose Online via ngrok

Open a second terminal:

ngrok http 5000


You’ll get a public URL like:

https://abcd1234.ngrok-free.app

🧪 How It Works (RAG Pipeline)

User uploads a PDF/DOCX

Text extracted & cleaned

Recursive text splitting

Embeddings generated (OpenAI/AWS)

Points stored in Pinecone

User asks question

Query embedding → Pinecone similarity search

Top chunks injected into LLM

Model answers based on document context

🛠️ Commands Overview
# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install packages
pip install -r requirements.txt

# Set environment variables
nano .env   # or any editor

# Run Flask server
python app.py

# Expose online
ngrok http 5000
