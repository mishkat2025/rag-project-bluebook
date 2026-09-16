# EWU RAG Question Answering System

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about East West University's Undergraduate Bulletin using **FAISS vector retrieval** and a locally running **Ollama LLM**.

The system retrieves relevant information from the EWU bulletin and generates grounded answers through a conversational terminal interface.

---

## 📌 Project Overview

University information is often distributed across lengthy academic bulletins, making it difficult for students to quickly find admission requirements, fees, academic regulations, and program information.

This project develops a local Retrieval-Augmented Generation system that allows users to ask questions in natural language and receive answers based on the East West University Undergraduate Bulletin.

### Main Objectives

* Build a document-based question-answering system.
* Retrieve relevant information using FAISS.
* Generate answers using a locally hosted Ollama LLM.
* Reduce unsupported answers through grounded prompting.
* Support continuous conversational interaction.
* Preserve document page references for retrieved information.
* Provide a modular foundation for future RAG experiments.

---

## 🏗️ System Architecture

```text
                    EWU Undergraduate Bulletin
                                │
                                ▼
                       PDF Text Extraction
                                │
                                ▼
                          Text Cleaning
                                │
                                ▼
                         Text Chunking
                                │
                                ▼
                       Embedding Model
                                │
                                ▼
                          FAISS Index
                                │
                                ▼
                       Saved Vector Store
                                │
                                │
                         User Question
                                │
                                ▼
                       Query Embedding
                                │
                                ▼
                       Hybrid Retrieval
                                │
                                ▼
                     Relevant PDF Chunks
                                │
                                ▼
                       Context Construction
                                │
                                ▼
                          Ollama LLM
                                │
                                ▼
                       Grounded Answer
                                │
                                ▼
                       Terminal Chatbot
```

### Core Technologies

| Component            | Technology                       |
| -------------------- | -------------------------------- |
| Programming Language | Python                           |
| Document Source      | EWU Undergraduate Bulletin       |
| Vector Database      | FAISS                            |
| Retrieval            | Dense + Phrase/Keyword Retrieval |
| LLM Runtime          | Ollama                           |
| LLM Model            | Qwen2.5:3B                       |
| Interface            | Terminal Chatbot                 |
| Environment          | Python Virtual Environment       |
| Development          | VS Code                          |

---

## ✨ Features

### 1. PDF-Based Question Answering

Ask questions about the EWU Undergraduate Bulletin, such as:

* What are the admission requirements for undergraduate programs?
* What are the requirements for B.Pharm?
* Is a written test required?
* What is the application fee?
* What are the graduation requirements?
* What courses are offered by the CSE department?

### 2. Hybrid FAISS Retrieval

The retrieval system combines semantic similarity with phrase-based matching to improve the selection of relevant document chunks.

```text
User Question
      │
      ├── Semantic Retrieval
      │
      └── Phrase Retrieval
                │
                ▼
          Combined Ranking
                │
                ▼
          Top-K Results
```

### 3. Local LLM Generation

The project uses Ollama to run Qwen2.5:3B locally.

```text
Python Application
       │
       ▼
Ollama API
       │
       ▼
Qwen2.5:3B
       │
       ▼
Generated Answer
```

No cloud LLM API is required for answer generation.

### 4. Continuous Chat

The chatbot supports multiple questions in one execution.

```text
You: What are the admission requirements for B.Pharm?

EWU Assistant: ...

You: What about Mathematics?

EWU Assistant: ...

You: Is a written test required?

EWU Assistant: ...

You: exit
```

### 5. Conversation History

Previous questions and answers are retained during the session to help interpret follow-up questions.

Available commands:

| Command | Description                |
| ------- | -------------------------- |
| `exit`  | End the chatbot            |
| `quit`  | End the chatbot            |
| `clear` | Clear conversation history |
| `q`     | Exit the chatbot           |

### 6. Grounded Answer Generation

The prompt instructs the LLM to:

* Use retrieved EWU bulletin content.
* Avoid inventing unsupported facts.
* Preserve exact values and conditions.
* Distinguish general admission from program-specific requirements.
* Distinguish application fees, admission fees, and tuition fees.
* State when the retrieved content is insufficient.

---

## 📂 Project Structure

```text
rag-project/
│
├── data/
│   ├── raw/
│   │   └── ewu_bulletin.pdf
│   │
│   └── processed/
│       ├── pdf_pages.json
│       └── pdf_chunks.json
│
├── vectorstore/
│   └── faiss/
│       ├── index.faiss
│       └── metadata.json
│
├── src/
│   ├── config.py
│   ├── retriever.py
│   └── ...
│
├── scripts/
│   ├── ask_rag.py
│   ├── test_retrieval.py
│   ├── test_ollama.py
│   └── ...
│
├── .venv/
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact file structure may vary as the project develops. The important components are the source code, document processing pipeline, FAISS index, metadata, and chatbot scripts.

---

## ⚙️ Installation

### Prerequisites

Install the following:

* Python 3.10 or newer
* Ollama
* Git
* VS Code (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/mishkat2025/rag-project-bluebook.git
```

Navigate into the project:

```bash
cd rag-project-bluebook
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🦙 Ollama Setup

Install Ollama from:

https://ollama.com/

Download the Qwen2.5 3B model:

```bash
ollama pull qwen2.5:3b
```

Test the model:

```bash
ollama run qwen2.5:3b
```

Example:

```text
>>> Explain what a university admission requirement is in one sentence.
```

Exit Ollama:

```text
/bye
```

### Verify Ollama Connection

Run:

```bash
python scripts/test_ollama.py
```

Expected output:

```text
============================================================
OLLAMA CONNECTION TEST
============================================================
Model: qwen2.5:3b

Generated answer:
...
```

---

## 🚀 Running the Chatbot

After activating the virtual environment:

```bash
python scripts/ask_rag.py
```

The chatbot will start:

```text
============================================================
EWU RAG QUESTION ANSWERING
============================================================
Type 'exit' or 'quit' to end the chat.
Type 'clear' to start a new conversation.
============================================================
```

### Example Conversation

```text
You: What are the admission requirements for B.Pharm?

Searching the EWU bulletin...
Retrieved 5 chunks.

EWU Assistant:
------------------------------------------------------------
The admission requirements for the B.Pharm program include:
...
------------------------------------------------------------
Retrieved source pages:
144, 143, 219, 177, 176

You: What about Mathematics?

Searching the EWU bulletin...
Retrieved 5 chunks.

EWU Assistant:
------------------------------------------------------------
...
------------------------------------------------------------

You: exit

Chat ended.
```

---

## 🔍 Testing FAISS Retrieval

To test the retrieval system independently:

```bash
python scripts/test_retrieval.py
```

Example:

```text
============================================================
HYBRID FAISS RETRIEVAL TEST
============================================================

Enter your question: What are the admission requirements for undergraduate students?

Retrieved results: 5
```

The retrieval test displays:

* Combined similarity score
* Semantic similarity score
* Phrase score
* PDF page number
* Chunk ID
* Retrieved text

This helps inspect whether the correct document passages are being retrieved.

---

## 🧠 How the RAG Pipeline Works

### Step 1 — Document Processing

The EWU Undergraduate Bulletin is processed into usable text.

```text
PDF
 ↓
Extract Text
 ↓
Clean Text
 ↓
Create Chunks
```

### Step 2 — Embedding Generation

Each document chunk is converted into a numerical vector using an embedding model.

```text
Document Chunk
      │
      ▼
Embedding Model
      │
      ▼
Numerical Vector
```

### Step 3 — FAISS Indexing

The vectors are stored in a FAISS index.

```text
Embeddings
    │
    ▼
FAISS Index
    │
    ▼
index.faiss
```

Document text and page metadata are stored separately so retrieved vectors can be mapped back to their original content.

### Step 4 — Query Retrieval

When the user asks a question:

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
FAISS Search
      │
      ▼
Relevant Chunks
```

### Step 5 — Context Construction

The retrieved chunks are assembled into a prompt containing:

* Source page numbers
* Chunk IDs
* Retrieved text
* User question
* Grounding instructions

### Step 6 — Answer Generation

The prompt is sent to Ollama.

```text
Retrieved Context
       +
User Question
       +
System Prompt
       │
       ▼
    Ollama
       │
       ▼
Generated Answer
```

---

## 🛡️ Grounding and Hallucination Reduction

The system uses a grounded prompting strategy to reduce unsupported answers.

The LLM is instructed to:

1. Use only the retrieved EWU bulletin context.
2. Avoid external knowledge.
3. Avoid guessing or inventing requirements.
4. Preserve exact numbers and conditions.
5. Distinguish general and program-specific information.
6. Report uncertainty when the context is insufficient.
7. Avoid confusing different types of fees.

### Example

**Question:**

```text
What is the minimum GPA required for undergraduate admission at EWU?
```

**System behavior:**

The assistant should retrieve the relevant admission section and answer from the supporting text, rather than relying on the model's general knowledge.

---

## 🧪 Current Testing

The system has been tested with questions related to:

| Test Category       | Example                              |
| ------------------- | ------------------------------------ |
| General Admission   | Undergraduate admission requirements |
| Program Admission   | B.Pharm requirements                 |
| Fees                | Application fee                      |
| Follow-up Questions | Mathematics requirements             |
| Admission Process   | Written test requirement             |
| Chat Functionality  | Multiple questions in one session    |
| Model Connection    | Ollama API connection                |
| Retrieval           | FAISS result inspection              |

### Important Limitation

The bulletin is a source document for the chatbot. A fee or policy mentioned in the bulletin may not necessarily be current.

For current admission fees, policies, and deadlines, users should verify the information with East West University.

---

## 📊 Research and Future Improvements

This project is designed to evolve beyond a basic chatbot into an experimental RAG system.

### Planned Improvements

* [ ] Improve conversational follow-up retrieval.
* [ ] Improve retrieval ranking.
* [ ] Add BM25 keyword retrieval.
* [ ] Compare dense, keyword, and hybrid retrieval.
* [ ] Evaluate Recall@K, Precision@K, MRR, and Hit Rate.
* [ ] Create a question-answer evaluation dataset.
* [ ] Evaluate answer faithfulness and relevance.
* [ ] Compare embedding models.
* [ ] Compare chunk sizes and overlap.
* [ ] Add a web-based interface.
* [ ] Add experiment logging and reproducibility metadata.

### Potential Research Direction

**Evaluation of Retrieval Strategies for Grounded Question Answering Using a Local LLM**

The system can be used to investigate how retrieval quality affects the correctness and faithfulness of generated answers.

---

## 📚 Dataset and Source

### Primary Document

East West University Undergraduate Bulletin, 14th Edition.

The document contains information about:

* University policies
* Undergraduate programs
* Admission requirements
* Department information
* Course descriptions
* Academic regulations
* Graduation requirements

### Source

East West University official website:

https://www.ewubd.edu/

The bulletin included in this project is used as the knowledge source for retrieval.

---

## ⚠️ Disclaimer

This project is an academic and experimental question-answering system.

The generated answers depend on the retrieved document content and the local language model. The system may occasionally retrieve incomplete or irrelevant passages.

This chatbot should not be treated as an official admission authority.

For official admission decisions, fees, deadlines, and policies, consult East West University.

---

## 👨‍💻 Author

**Md. Saiful Islam**



---

## 📄 License

This project is intended for educational and research purposes.

Add an appropriate open-source license before publishing the repository.

---

## ⭐ Acknowledgments

* East West University — Bulletin source
* FAISS — Vector similarity search
* Ollama — Local LLM runtime
* Qwen — Language model
* Python — Project implementation
