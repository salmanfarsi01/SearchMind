import os
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# -------------------
# CONFIG
# -------------------
load_dotenv()
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-chat-index")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536 dims
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4.1-mini")

UPLOAD_FOLDER = os.path.join("data", "uploads")
MANIFEST_FOLDER = os.path.join("data", "manifests")
ALLOWED_EXTENSIONS = {"pdf", "docx"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MANIFEST_FOLDER, exist_ok=True)

# Set your keys as environment variables before running:
# export OPENAI_API_KEY="..."
# export PINECONE_API_KEY="..."

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


def get_or_create_index():
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",  # change if needed
            ),
        )
    return pc.Index(PINECONE_INDEX_NAME)


index = get_or_create_index()


# -------------------
# UTILS
# -------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(texts)


def extract_text_from_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def recursive_chunk(
    text: str,
    max_size: int = 1000,
    overlap: int = 200,
    separators: List[str] = None
) -> List[str]:
    """
    Simple recursive character splitter similar to LangChain's RecursiveCharacterTextSplitter.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def split_with_sep(t: str, seps: List[str]) -> List[str]:
        if len(t) <= max_size:
            return [t]

        if not seps:
            # no more separators, just hard cut
            chunks = []
            start = 0
            while start < len(t):
                end = min(start + max_size, len(t))
                chunks.append(t[start:end])
                start = end - overlap  # with overlap
            return chunks

        sep = seps[0]
        parts = t.split(sep)
        current = ""
        pieces = []
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                # if single part is too big, go deeper
                if len(part) > max_size:
                    sub_pieces = split_with_sep(part, seps[1:])
                    pieces.extend(sub_pieces)
                    current = ""
                else:
                    current = part
        if current:
            pieces.append(current)

        # apply overlap
        final_chunks = []
        for i, chunk in enumerate(pieces):
            if i == 0:
                final_chunks.append(chunk)
            else:
                start_overlap = max(0, len(chunk) - overlap)
                prev_tail = chunk[start_overlap:]
                merged = prev_tail + chunk
                # to keep it simple, just append chunk
                final_chunks.append(chunk)
        return pieces

    # Normalize whitespace
    text = " ".join(text.split())
    chunks = split_with_sep(text, separators)

    # final pass to enforce max_size & overlap strictly
    final = []
    for ch in chunks:
        if len(ch) <= max_size:
            final.append(ch)
        else:
            start = 0
            while start < len(ch):
                end = min(start + max_size, len(ch))
                final.append(ch[start:end])
                start = end - overlap
    # Strip and filter empty
    return [c.strip() for c in final if c.strip()]


def get_embedding(text: str) -> List[float]:
    """
    If you're using AWS instead of OpenAI directly, replace this
    implementation with your AWS SDK call, but keep the return as a
    1536-dim float list.
    """
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]
    )
    return resp.data[0].embedding


def chat_with_context(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n".join(
        [f"[Chunk {i+1}]\n{ch}" for i, ch in enumerate(context_chunks)]
    )
    system_prompt = (
        "You are a helpful RAG assistant. Answer the user's question using ONLY "
        "the provided document chunks as factual source. "
        "If the answer is not in the context, say you don't know."
        "**Must provide the answer based on user's provided document chunk**"
        "DO NOT GIVE ANY IRRELEVANT ANSWERS."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context_text}\n\n"
                f"QUESTION: {question}\n\n"
                "Answer in a clear, concise way."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def save_manifest(doc_id: str, meta: Dict[str, Any]):
    manifest_path = os.path.join(MANIFEST_FOLDER, f"{doc_id}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_manifest(doc_id: str) -> Dict[str, Any]:
    manifest_path = os.path.join(MANIFEST_FOLDER, f"{doc_id}.json")
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------
# FLASK APP
# -------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    doc_id = str(uuid.uuid4())

    stored_filename = f"{doc_id}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)
    file.save(file_path)

    # Extract text
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == "docx":
        text = extract_text_from_docx(file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    if not text.strip():
        return jsonify({"error": "No text found in file"}), 400

    # Chunk and embed
    chunks = recursive_chunk(text, max_size=1000, overlap=200)
    vectors = []
    for i, ch in enumerate(chunks):
        vec_id = f"{doc_id}_chunk_{i}"
        embedding = get_embedding(ch)
        vectors.append(
            {
                "id": vec_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": ch[:5000],  # store part of text
                },
            }
        )

    if vectors:
        index.upsert(vectors=vectors)

    manifest = {
        "doc_id": doc_id,
        "filename": filename,
        "stored_filename": stored_filename,
        "num_chunks": len(chunks),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
    }
    save_manifest(doc_id, manifest)

    return jsonify(
        {
            "message": "File uploaded and indexed successfully",
            "doc_id": doc_id,
            "filename": filename,
            "num_chunks": len(chunks),
        }
    )


@app.route("/api/documents", methods=["GET"])
def list_docs():
    manifests = []
    for fname in os.listdir(MANIFEST_FOLDER):
        if fname.endswith(".json"):
            with open(os.path.join(MANIFEST_FOLDER, fname), "r", encoding="utf-8") as f:
                manifests.append(json.load(f))
    return jsonify(manifests)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = data.get("message") or ""
    doc_ids = data.get("doc_ids")  # optional list

    if not question.strip():
        return jsonify({"error": "Message is required"}), 400

    # Embed query
    query_embedding = get_embedding(question)

    # Build filter for selected docs (if any)
    pinecone_filter = None
    if doc_ids:
        pinecone_filter = {"doc_id": {"$in": doc_ids}}

    res = index.query(
        vector=query_embedding,
        top_k=6,
        include_metadata=True,
        filter=pinecone_filter,
    )

    context_chunks = []
    context_sources = []
    for match in res.matches:
        meta = match.metadata or {}
        text = meta.get("text", "")
        context_chunks.append(text)
        context_sources.append(
            {
                "doc_id": meta.get("doc_id"),
                "filename": meta.get("filename"),
                "chunk_index": meta.get("chunk_index"),
                "score": match.score,
            }
        )

    if not context_chunks:
        return jsonify(
            {
                "answer": "I couldn't find relevant information in the uploaded documents.",
            }
        )

    answer = chat_with_context(question, context_chunks)

    return jsonify(
        {
            "answer": answer,

        }
    )


if __name__ == "__main__":
    app.run(debug=True)
