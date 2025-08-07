 💼 RAG Pipeline for Financial Data Question Answering

 This project was developed as part of the **Sigmoix.AI AI R&D Engineer Intern Assessment**. It implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer natural language questions using financial documents, specifically **Meta’s Q1 2024 Financial Report**.



📌 Objective

Design and implement a Retrieval-Augmented Generation (RAG) system that:
- Extracts and processes financial text and tables from a real-world PDF document.
- Embeds both structured and unstructured data for semantic retrieval.
- Uses a Large Language Model (LLM) to generate natural language answers to user queries.
- Incorporates reranking and query optimization for better relevance and accuracy.
- --------------------------------------------------------------------------------
📄 How It Works
🔹 Step 1: Text and Table Extraction
Extracts raw text using PyPDF2

Extracts tabular data using Camelot (stream flavor)

🔹 Step 2: Embedding & Context Preparation
Chunks raw text and prepares markdown representation of tables

Encodes both using SentenceTransformer for similarity matching

🔹 Step 3: Query Optimization & Retrieval
Optimizes the user query using rule-based logic

Retrieves top-k relevant contexts (text or tables) using cosine similarity

Optionally reranks the results with a cross-encoder

🔹 Step 4: Answer Generation
Prompts a text generation model (distilgpt2) using retrieved context

Post-processes the output to remove hallucinations and self-introductions

🔍 Example Queries
Here are some example queries the pipeline can handle:

✅ What was Meta's revenue in Q1 2024?

✅ How much did Meta spend on capital expenditures?

✅ What is Meta's headcount as of March 31, 2024?

✅ Summarize operating expenses for Q1 2024.

✅ Compare Meta's net income in Q1 2024 vs Q1 2023.



