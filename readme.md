EWU Advanced RAG Chatbot

A local, evidence-grounded Retrieval-Augmented Generation (RAG) chatbot for answering questions from the East West University (EWU) Undergraduate Bulletin.

The project combines structure-aware document ingestion, hybrid retrieval, re-ranking, and a controlled multi-agent workflow. The LLM-based agents use a local Ollama-hosted Qwen model, while embeddings, keyword retrieval, vector storage, and re-ranking are handled by dedicated components.

Knowledge source: EWU Undergraduate Bulletin, 14th Edition (November 2019).

The bulletin itself notes that EWU may change policies, fees, curricula, and other information and that the publication is not a contract or guarantee. Therefore, answers produced by this system should be understood as answers grounded in the indexed bulletin, not as a replacement for current official university information.

1. Project Overview

Traditional RAG systems often follow a simple pipeline:

Question
   ↓
Retrieve documents
   ↓
Generate answer

This project uses a more controlled architecture:

EWU Bulletin PDF
       ↓
Document Ingestion
       ↓
Structure Analysis
       ↓
Structure-Aware Chunking
       ↓
Metadata + Validation
       ↓
 ┌───────────────┐
 │ ChromaDB      │
 │ BM25 Index    │
 └───────────────┘
       ↓
Supervisor Agent
       ↓
Query Planning
       ↓
Hybrid Retrieval
       ↓
Re-ranking
       ↓
Evidence Selection
       ↓
Answer Generation
       ↓
Final Verification
       ↓
Grounded Answer + Page Sources

The goal is to reduce unsupported answers by making retrieval, evidence selection, answer generation, and verification explicit stages of the workflow.

2. Key Features

Document processing

Page-by-page PDF extraction

Original page-number preservation

Document structure analysis

Structure-aware chunking

Section and heading metadata

Program-specific metadata where available

Cleaned and validated chunks

Retrieval

Dense semantic retrieval

BM25 keyword retrieval

Hybrid retrieval

Candidate fusion

Re-ranking

Metadata-aware evidence

Multi-agent workflow

The system contains seven logical agents:

Supervisor Agent

Query Planning Agent

Retrieval Agent

Re-ranking Agent

Evidence Agent

Answer Agent

Final Verification Agent

The agents operate through shared workflow state rather than unrestricted agent-to-agent communication.

Local LLM

The LLM-based agents use the same locally hosted Ollama/Qwen model with different prompts and responsibilities.

Ollama / Qwen
    ├── Supervisor
    ├── Query Planning
    ├── Evidence
    ├── Answer
    └── Verification

Non-LLM components remain specialized:

Sentence Transformer → embeddings
BM25                → keyword retrieval
ChromaDB            → vector storage
Re-ranker           → relevance scoring
Ollama/Qwen         → reasoning and generation

3. Architecture

High-Level Architecture

                         ┌─────────────────────┐
                         │  EWU Bulletin PDF   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   PDF Parser        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Structure Analyzer  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Structure-Aware     │
                         │ Chunker              │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Metadata + Cleaning │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
             ┌─────────────┐                 ┌─────────────┐
             │  ChromaDB   │                 │    BM25     │
             │ Dense Index │                 │ Keyword     │
             └──────┬──────┘                 └──────┬──────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Supervisor Agent    │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Query Planning      │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Hybrid Retrieval    │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Re-ranking          │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Evidence Agent      │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Answer Agent        │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Verification Agent  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Final Answer        │
                         │ + Page References   │
                         └─────────────────────┘

4. Multi-Agent Workflow

4.1 Supervisor Agent

The Supervisor is responsible for controlling the workflow.

It determines whether the question is:

simple,

complex,

requires query planning,

requires retrieval,

requires retry,

ready for answer generation, or

ready for final verification.

The Supervisor does not directly answer the user's question.

4.2 Query Planning Agent

For complex questions, the Query Planning Agent decomposes the original question into useful retrieval queries or sub-questions.

For example:

Question:
Compare the graduation requirements of CSE and B.Pharm.

Possible sub-questions:
1. What are the CSE graduation requirements?
2. What are the B.Pharm graduation requirements?

The resulting queries are passed to the retrieval stage.

4.3 Retrieval Agent

The Retrieval Agent searches the indexed bulletin using:

dense semantic retrieval,

BM25 keyword retrieval,

hybrid fusion.

The objective is to retrieve a sufficiently broad candidate set before re-ranking.

4.4 Re-ranking Agent

The re-ranking stage scores retrieved candidates for relevance to the user's question.

Its responsibility is only:

Find the most relevant evidence.

It does not generate the final answer.

4.5 Evidence Agent

The Evidence Agent evaluates the retrieved candidates and selects the evidence that can support the answer.

It can also identify missing information.

The architecture uses a bounded retry strategy so that retrieval does not continue indefinitely.

Conceptually:

Evidence sufficient?
       │
   ┌───┴───┐
  YES      NO
   │        │
   ▼        ▼
 Answer   Retry retrieval
             │
             ▼
        Retry limit reached
             │
             ▼
      Controlled limitation

The planned maximum retrieval retry limit is 2.

4.6 Answer Agent

The Answer Agent generates the final response from the selected bulletin evidence.

It is instructed to:

Use only retrieved bulletin content.

Avoid inventing facts.

Answer all relevant parts of the question.

Preserve exact numbers and requirements.

Include page information.

Clearly identify missing information.

Use tables when appropriate for comparisons.

4.7 Final Verification Agent

The Verification Agent performs a final consistency check.

It checks:

whether major claims are supported,

whether cited pages are appropriate,

whether all sub-questions were addressed,

whether information from unrelated programs was mixed,

whether outside knowledge was introduced,

whether important evidence is missing.

A successful verification can return a structure such as:

{
  "approved": true,
  "unsupported_claims": [],
  "missing_subquestions": [],
  "citation_errors": [],
  "outside_knowledge": [],
  "notes": []
}

5. Project Structure

The project is organized into separate layers for ingestion, storage, retrieval, agents, orchestration, and generation.

rag-project/
│
├── data/
│   ├── raw/
│   │   └── ewu_bulletin.pdf
│   │
│   ├── processed/
│   │   ├── chunks.json
│   │   └── metadata.json
│   │
│   └── indexes/
│       ├── chroma/
│       └── bm25/
│
├── src/
│   │
│   ├── ingestion/
│   │   ├── pdf_parser.py
│   │   ├── structure_analyzer.py
│   │   ├── chunker.py
│   │   └── metadata_builder.py
│   │
│   ├── storage/
│   │   ├── vector_store.py
│   │   ├── chroma_store.py
│   │   └── trace_store.py
│   │
│   ├── retrieval/
│   │   ├── dense_retriever.py
│   │   ├── bm25_retriever.py
│   │   ├── hybrid_retriever.py
│   │   ├── fusion.py
│   │   └── reranker.py
│   │
│   ├── agents/
│   │   ├── supervisor_agent.py
│   │   ├── query_planning_agent.py
│   │   ├── retrieval_agent.py
│   │   ├── evidence_agent.py
│   │   ├── answer_agent.py
│   │   └── verification_agent.py
│   │
│   ├── orchestration/
│   │   ├── workflow.py
│   │   ├── state.py
│   │   └── routing.py
│   │
│   ├── generation/
│   │   ├── ollama_client.py
│   │   └── prompts.py
│   │
│   └── config/
│       └── settings.py
│
├── scripts/
│   ├── build_index.py
│   ├── test_retrieval.py
│   ├── test_routing.py
│   └── test_workflow_suite.py
│
├── tests/
│
├── app/
│   └── chat.py
│
├── .env
├── .env.example
├── requirements.txt
├── README.md
└── ...

6. Technology Stack

Component

Technology

Language

Python

PDF processing

PyMuPDF

Embeddings

Sentence Transformers

Vector database

ChromaDB

Keyword retrieval

BM25

Retrieval

Dense + BM25 hybrid

Re-ranking

Dedicated re-ranking component

LLM

Local Ollama / Qwen

Configuration

Pydantic / pydantic-settings

HTTP client

Requests

Testing

Pytest

Environment variables

python-dotenv

The project is designed so that the retrieval/storage layers remain separated from the agent logic.

7. Requirements

Recommended Python dependencies include:

PyMuPDF
sentence-transformers
chromadb
rank-bm25
pydantic
pydantic-settings
requests
numpy
scikit-learn
python-dotenv
pytest

If the current re-ranker implementation uses a separate package, install the package required by the actual implementation in src/retrieval/reranker.py.

8. Installation

8.1 Clone the project

git clone <your-repository-url>
cd rag-project

8.2 Create a virtual environment

Windows PowerShell:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

If PowerShell blocks script execution, the virtual environment can also be activated using the appropriate Windows command shell or by adjusting the execution policy for the current user/environment.

Linux/macOS:

python3 -m venv .venv
source .venv/bin/activate

8.3 Install dependencies

pip install -r requirements.txt

9. Ollama Setup

The chatbot uses a locally hosted Ollama model.

Install Ollama and make sure the selected Qwen model is available locally.

Verify Ollama is running:

ollama list

Then verify the model configured by the project is available.

The exact model name should match the value configured in the project's settings/environment configuration.

The application communicates with Ollama through the project's ollama_client.py.

10. Environment Configuration

Create a .env file from .env.example:

cp .env.example .env

On Windows PowerShell:

Copy-Item .env.example .env

Configure the values required by the project.

Typical configuration includes:

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=<configured-qwen-model>

Use the actual variable names defined in src/config/settings.py and .env.example.

Do not commit private tokens or credentials to Git.

11. Document Ingestion

Place the EWU bulletin PDF in:

data/raw/ewu_bulletin.pdf

The ingestion process follows:

PDF
 ↓
Page-by-page extraction
 ↓
Structure analysis
 ↓
Structure-aware chunking
 ↓
Metadata generation
 ↓
Cleaning / validation
 ↓
Embeddings
 ↓
ChromaDB
 ↓
BM25

The resulting data is stored under:

data/
├── processed/
│   ├── chunks.json
│   └── metadata.json
│
└── indexes/
    ├── chroma/
    └── bm25/

Build or rebuild the indexes using the project's indexing script:

python scripts/build_index.py

12. Running the Chatbot

Once the indexes have been built and Ollama is running, start the application using the project's chat entry point:

python app/chat.py

If the project is being run through the existing script-based interface, use the corresponding chat script configured in the repository.

The interactive chatbot supports conversational questions over the EWU bulletin.

Example:

You: What is the admission requirement for B.Pharm?

The system then performs the controlled workflow:

Question
  ↓
Supervisor
  ↓
Query Planning
  ↓
Retrieval
  ↓
Re-ranking
  ↓
Evidence
  ↓
Answer
  ↓
Verification
  ↓
Response

13. Example Questions

The system is intended for questions whose answers can be grounded in the EWU Undergraduate Bulletin.

Examples:

What is the admission requirement for B.Pharm?

How many credits are required for the B.Pharm degree?

What are the admission requirements for a particular undergraduate program?

What courses are included in a specific program?

What are the graduation requirements for a program?

Compare the requirements of two programs.

For questions requiring information that is not present in the indexed bulletin, the system should provide a controlled limitation rather than inventing an answer.

Example:

The available EWU Undergraduate Bulletin evidence is
insufficient to answer this question.

14. Retrieval Strategy

The project uses hybrid retrieval because semantic and keyword retrieval provide different strengths.

Dense retrieval

Dense embeddings help retrieve passages that are semantically related even when the exact wording differs.

Question
   ↓
Embedding
   ↓
Vector search
   ↓
Semantic candidates

BM25 retrieval

BM25 provides lexical matching and is useful for:

exact terminology,

program names,

course codes,

numbers,

requirements,

specific phrases.

Question
   ↓
Tokenization
   ↓
BM25
   ↓
Keyword candidates

Hybrid retrieval

The two candidate sets are combined before re-ranking:

Dense candidates ─┐
                  ├──→ Fusion → Re-ranking → Evidence
BM25 candidates ──┘

15. Structure-Aware Chunking

Instead of treating the PDF as one continuous text stream, the ingestion pipeline preserves document structure where practical.

A chunk can contain metadata such as:

{
  "chunk_id": "ewu-p177-c00296",
  "page": 177,
  "source": "ewu_undergraduate_bulletin.pdf",
  "section": "Faculty of Pharmacy",
  "heading": "Admission Requirements",
  "program": "B.Pharm",
  "content_type": "paragraph",
  "chunk_position": 3
}

This allows retrieved evidence to retain its context and original page location.

16. Evidence Grounding

The central design principle is:

The chatbot should answer from the retrieved EWU bulletin evidence rather than relying on general model knowledge.

The Answer Agent is therefore not intended to independently research the internet or invent missing information.

The evidence flow is:

Retrieved Candidates
       ↓
Re-ranking
       ↓
Evidence Selection
       ↓
Supported Context
       ↓
Answer Generation
       ↓
Verification

This makes it easier to inspect why a particular answer was generated.

17. Retrieval Traceability

The architecture includes a retrieval trace layer for recording useful workflow information.

A trace can contain information such as:

Original question
Rewritten queries
Subqueries
Retrieved chunks
Re-ranking results
Retry count
Evidence decision
Final verification result

This is useful for:

debugging retrieval failures,

evaluating agent behavior,

understanding why an answer was produced,

improving prompts,

comparing retrieval strategies.

18. Testing

The project includes tests for individual stages and the overall workflow.

Routing test

python -m scripts.test_routing

The routing test verifies the expected agent sequence:

supervisor
→ query_planning
→ retrieval
→ reranking
→ evidence
→ answer
→ verification

Workflow test suite

python -m scripts.test_workflow_suite

The workflow suite tests different question types and checks the resulting workflow state.

Retrieval test

python scripts/test_retrieval.py

Use this to inspect retrieval and re-ranking behavior for individual questions.

19. Example Workflow Result

For a question such as:

What is the admission requirement for B.Pharm?

the system can produce a workflow state containing:

Workflow:
simple

Agents:
supervisor
query_planning
retrieval
reranking
evidence
answer
verification

Retries:
0

Evidence:
sufficient

Verification:
approved

The final answer is generated from the selected bulletin evidence.

20. Failure Handling

The system is designed to fail conservatively.

If retrieval does not provide enough useful evidence, the system should not fabricate a response.

Instead, it can return a controlled response indicating that the available bulletin evidence is insufficient.

The architecture also limits retrieval retries:

Maximum retrieval retries = 2

This prevents endless retrieval loops.

21. Design Principles

Grounded generation

Answers should be supported by retrieved bulletin evidence.

Separation of concerns

Each component has a specific responsibility:

Ingestion       → understand and index the document
Retrieval       → find candidate evidence
Re-ranking      → prioritize relevant candidates
Evidence        → determine support
Answer          → generate response
Verification    → check the final response

Controlled orchestration

The Supervisor controls workflow transitions rather than allowing unrestricted agent communication.

Local-first operation

The project is designed to run the LLM locally through Ollama.

Traceability

Important retrieval and verification information can be recorded for debugging and evaluation.

Bounded retries

The system avoids infinite retrieval or agent loops.

22. Current Limitations

The project should not be treated as a perfect information-retrieval system.

Potential limitations include:

PDF extraction can introduce formatting artifacts.

Tables and complex layouts can be difficult to parse perfectly.

Retrieval may occasionally rank related but less relevant passages highly.

Re-ranking quality depends on the selected model/component.

A local LLM can still produce unsupported wording if prompts and verification are insufficient.

Historical bulletin information may no longer represent current EWU policy.

The indexed bulletin is limited to the information contained in the source document.

Because the bulletin is from 2019, current university policies should be checked against official EWU sources before being relied upon for real-world decisions.

23. Future Improvements

Possible future work includes:

Better table extraction

More advanced structure detection

Improved program-aware retrieval

Metadata filtering before semantic retrieval

Better query expansion

Improved re-ranking

Retrieval evaluation using Recall@K and MRR

Answer faithfulness evaluation

Citation precision evaluation

Automated regression tests

Web or API interface

Conversation/session persistence

Improved retrieval trace visualization

More sophisticated verification and answer regeneration

Evaluation against a manually created EWU question-answer benchmark

24. Research and Academic Value

This project can be used as a practical demonstration of:

Retrieval-Augmented Generation

Multi-agent orchestration

Hybrid information retrieval

Semantic search

BM25 retrieval

Vector databases

Re-ranking

Evidence-grounded generation

LLM verification

Local LLM deployment

Document intelligence

The architecture also provides a useful experimental platform for evaluating whether adding controlled agent stages improves retrieval and answer reliability compared with a simpler single-stage RAG pipeline.

25. Source Document

The primary knowledge source is:

East West University
Undergraduate Bulletin
14th Edition
November 2019

The chatbot should treat this document as its authoritative knowledge base for indexed questions, while recognizing that the bulletin itself may not represent current university policy.

26. License

Add the project's intended license here, for example:

MIT License

or replace this section with the license required by your institution/project.

27. Author

EWU Advanced RAG Project

Built as an academic and practical project demonstrating an evidence-grounded, local multi-agent RAG architecture.

28. Quick Start

For a concise setup:

# 1. Create environment
python -m venv .venv

# 2. Activate environment
# Windows:
.\.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama
ollama list

# 5. Put the bulletin here
# data/raw/ewu_bulletin.pdf

# 6. Build indexes
python scripts/build_index.py

# 7. Run tests
python -m scripts.test_routing
python -m scripts.test_workflow_suite

# 8. Start the chatbot
python app/chat.py

29. Architecture Summary

                    EWU BULLETIN
                         │
                         ▼
              ┌─────────────────────┐
              │  INGESTION PIPELINE │
              └──────────┬──────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     ChromaDB                         BM25
   Dense Retrieval              Keyword Retrieval
          │                             │
          └──────────────┬──────────────┘
                         ▼
                 Hybrid Retrieval
                         │
                         ▼
                    Re-ranking
                         │
                         ▼
                  Evidence Agent
                         │
                         ▼
                    Answer Agent
                         │
                         ▼
                Verification Agent
                         │
                         ▼
              GROUNDED FINAL ANSWER
                    + PAGE SOURCES
                    
Core principle:

Retrieve evidence first, generate from evidence, and verify the generated answer before returning it