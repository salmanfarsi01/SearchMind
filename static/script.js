const uploadForm = document.getElementById("upload-form");
const uploadStatus = document.getElementById("upload-status");
const uploadButtonText = document.getElementById("upload-button-text");
const uploadSpinner = document.getElementById("upload-spinner");
const docsList = document.getElementById("documents-list");
const refreshDocsBtn = document.getElementById("refresh-docs");

const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatWindow = document.getElementById("chat-window");
const chatStatus = document.getElementById("chat-status");
const chatSendBtn = document.getElementById("chat-send-btn");

let documents = [];
let selectedDocIds = new Set();

// ------------- Helpers -------------

function createDocRow(doc) {
  const row = document.createElement("div");
  row.className =
    "flex items-start justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/70 px-3 py-2";

  const left = document.createElement("div");
  left.className = "flex-1";

  const title = document.createElement("div");
  title.className = "text-[11px] font-semibold text-slate-100";
  title.textContent = doc.filename;

  const meta = document.createElement("div");
  meta.className = "text-[10px] text-slate-400 mt-0.5";
  meta.textContent = `id: ${doc.doc_id} · chunks: ${doc.num_chunks} · uploaded: ${new Date(
    doc.uploaded_at
  ).toLocaleString()}`;

  left.appendChild(title);
  left.appendChild(meta);

  const right = document.createElement("div");
  right.className = "flex items-center gap-1";

  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.className = "h-4 w-4 accent-indigo-500";
  checkbox.checked = selectedDocIds.size === 0 || selectedDocIds.has(doc.doc_id);

  checkbox.addEventListener("change", () => {
    if (checkbox.checked) {
      selectedDocIds.add(doc.doc_id);
    } else {
      selectedDocIds.delete(doc.doc_id);
    }
  });

  right.appendChild(checkbox);

  row.appendChild(left);
  row.appendChild(right);
  return row;
}

async function loadDocuments() {
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) throw new Error("Failed to load documents");
    documents = await res.json();

    docsList.innerHTML = "";
    if (!documents.length) {
      docsList.innerHTML =
        '<p class="text-slate-500 text-xs italic">No documents uploaded yet.</p>';
      return;
    }

    documents
      .sort((a, b) => new Date(b.uploaded_at) - new Date(a.uploaded_at))
      .forEach((doc) => {
        const row = createDocRow(doc);
        docsList.appendChild(row);
      });
  } catch (err) {
    console.error(err);
    docsList.innerHTML =
      '<p class="text-red-400 text-xs">Error loading documents.</p>';
  }
}

function appendMessage(role, text) {
  const wrapper = document.createElement("div");
  const isUser = role === "user";
  wrapper.className = `flex ${
    isUser ? "justify-end" : "justify-start"
  } text-xs`;

  const bubble = document.createElement("div");
  bubble.className =
    "max-w-[80%] rounded-xl px-3 py-2 whitespace-pre-wrap shadow-sm border";

  if (isUser) {
    bubble.classList.add(
      "bg-indigo-500",
      "border-indigo-400",
      "text-slate-50"
    );
  } else {
    bubble.classList.add(
      "bg-slate-900",
      "border-slate-700",
      "text-slate-100"
    );
  }

  bubble.textContent = text;
  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ------------- Events -------------

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  uploadStatus.textContent = "";
  const fileInput = document.getElementById("file");
  if (!fileInput.files.length) {
    uploadStatus.textContent = "Please choose a file first.";
    uploadStatus.className = "mt-2 text-xs text-amber-300";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  uploadButtonText.textContent = "Indexing...";
  uploadSpinner.classList.remove("hidden");
  uploadStatus.textContent = "";
  uploadStatus.className = "mt-2 text-xs text-slate-300";

  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Upload failed");
    }

    uploadStatus.textContent = `Uploaded "${data.filename}" with ${data.num_chunks} chunks.`;
    uploadStatus.className = "mt-2 text-xs text-emerald-300";

    fileInput.value = "";
    await loadDocuments();
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = err.message || "Upload error.";
    uploadStatus.className = "mt-2 text-xs text-red-400";
  } finally {
    uploadButtonText.textContent = "Upload & Index";
    uploadSpinner.classList.add("hidden");
  }
});

refreshDocsBtn.addEventListener("click", async () => {
  await loadDocuments();
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  chatStatus.textContent = "";

  const text = chatInput.value.trim();
  if (!text) return;

  appendMessage("user", text);
  chatInput.value = "";
  chatInput.style.height = "auto";

  chatSendBtn.disabled = true;
  chatStatus.textContent = "Thinking...";
  chatStatus.className = "mt-1 text-[11px] text-slate-400";

  try {
    const body = { message: text };
    const activeDocIds = Array.from(selectedDocIds);
    if (activeDocIds.length) {
      body.doc_ids = activeDocIds;
    }

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Chat request failed");
    }

    appendMessage("assistant", data.answer || "[No answer returned]");

    if (data.sources && data.sources.length) {
      const srcText =
        "Sources:\n" +
        data.sources
          .map(
            (s) =>
              `• ${s.filename} (chunk ${s.chunk_index}, score: ${s.score.toFixed(
                3
              )})`
          )
          .join("\n");
      appendMessage("assistant", srcText);
    }

    chatStatus.textContent = "";
  } catch (err) {
    console.error(err);
    appendMessage("assistant", "Error: " + (err.message || "Something went wrong."));
    chatStatus.textContent = "There was an error answering your question.";
    chatStatus.className = "mt-1 text-[11px] text-red-400";
  } finally {
    chatSendBtn.disabled = false;
  }
});

// Auto-load docs on page load
document.addEventListener("DOMContentLoaded", () => {
  loadDocuments();
});

const fileInput = document.getElementById("file");
const selectedFileText = document.getElementById("selected-file");

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        selectedFileText.textContent = `Selected: ${fileInput.files[0].name}`;
        selectedFileText.classList.remove("hidden");
    } else {
        selectedFileText.textContent = "";
        selectedFileText.classList.add("hidden");
    }
});

