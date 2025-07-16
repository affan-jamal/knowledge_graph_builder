# 🧠 Knowledge Graph Builder (Neo4j + Hugging Face)

Build a local semantic search graph by generating embeddings for text, storing them in Neo4j, and performing vector-based similarity queries — without using paid APIs.

---

## 🚀 Overview

This tool demonstrates a local, lightweight Retrieval-Augmented Generation (RAG)-style pipeline using:

- 🐍 Python
- 🌐 Neo4j Graph Database
- 🤗 Hugging Face SentenceTransformer (`all-MiniLM-L6-v2`)

The system ingests text, generates local embeddings, stores them in a graph, and links similar entries using cosine similarity.

---

## 💼 Context: Task for pmspace.ai

This project was completed as a response to a technical task from **pmspace.ai**, for a role involving:

- Graph databases (Neo4j)
- RAG pipelines
- Embedding-based retrieval
- High-performance language model inference

While the original task specified OpenAI’s API, this version uses **open-source alternatives** due to API access constraints.

---

## 🔧 Features

- Generate sentence embeddings using Hugging Face locally (no API key required)
- Store sentences and vector data in Neo4j
- Build `:SIMILAR` edges between semantically related nodes
- Run vector similarity queries using Cypher

---

## 🛠️ Tech Stack

| Tool/Lib             | Role                            |
|----------------------|----------------------------------|
| Python 3.8+          | Core language                   |
| sentence-transformers| Local embedding model           |
| Neo4j                | Graph database                  |
| NumPy                | Cosine similarity               |
| dotenv               | Credential management           |

---

## 📝 How to Run

### 🔹 Install requirements

```bash
pip install -r requirements.txt
```

## Add .env file

- NEO4J_URI=bolt://localhost:7687
- NEO4J_USER=neo4j
- NEO4J_PASSWORD=your_password


## 🧪 Example Output

### Stored nodes:
- Python is a versatile programming language.
- Neo4j excels at storing and traversing relationships.
- Embeddings capture the semantics of a sentence.

### Query ➜ Which database treats relationships as first-class data?
- 0.887 → Neo4j excels at storing and traversing relationships.
- 0.471 → Embeddings capture the semantics of a sentence.
- 0.455 → Python is a versatile programming language.

