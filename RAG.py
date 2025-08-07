import PyPDF2
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from transformers import pipeline
import numpy as np
import camelot
import re

# --- Step 1: Basic RAG Pipeline ---

# 1.1 Text Extraction from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    # Basic cleaning: replace multiple newlines with single space, remove extra spaces
                    page_text = re.sub(r'\n+', ' ', page_text)
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    text += page_text + " "
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# 1.2 Text Chunking
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    chunks = []
    current_pos = 0
    while current_pos < len(text):
        end_pos = min(current_pos + chunk_size, len(text))
        chunk = text[current_pos:end_pos]
        chunks.append(chunk)
        if end_pos == len(text):
            break
        current_pos += (chunk_size - chunk_overlap)
    return chunks

# 1.3 Embedding Model Loading
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return None

# 1.4 LLM for Generation
def load_generator_llm(model_name="distilgpt2"):
    try:
        # Using 'text-generation' pipeline for simplicity with distilgpt2
        generator = pipeline("text-generation", model=model_name)
        return generator
    except Exception as e:
        print(f"Error loading generator LLM: {e}")
        return None

# --- Step 2: Structured Data Integration ---

# 2.1 Table Extraction using Camelot
def extract_tables_from_pdf(pdf_path, pages='all'):
    tables = []
    try:
        # flavor='stream' is often good for tables without clear lines, 'lattice' for ruled tables
        # Adjust table_areas if specific regions are known for tables
        extracted_tables = camelot.read_pdf(pdf_path, pages=pages, flavor='stream', line_scale=40)
        for i, table in enumerate(extracted_tables):
            df = table.df
            # Add a context/source for the table
            table_context = f"Table {i+1} from page {table.page}: {df.to_markdown(index=False)}"
            tables.append({'dataframe': df, 'markdown': df.to_markdown(index=False), 'context_text': table_context})
        print(f"Successfully extracted {len(tables)} tables.")
    except Exception as e:
        print(f"Error extracting tables: {e}")
    return tables

# 2.2 Hybrid Retrieval Helper: Combine Text and Table Context
# This function would be integrated into your overall retrieval logic.
# For simplicity, this example just prepares text for embedding.
def prepare_context_for_embedding(text_chunks, tables):
    all_contexts = []
    for chunk in text_chunks:
        all_contexts.append({'type': 'text', 'content': chunk})
    for table in tables:
        # Create a text representation of the table for embedding
        # This could be a summary, or the markdown itself
        table_text_representation = f"Financial Table: {table['markdown']}"
        all_contexts.append({'type': 'table', 'content': table_text_representation, 'original_df': table['dataframe']})
    return all_contexts

# Example of how you might perform a keyword lookup on tables
def find_in_tables_by_keywords(tables, keywords):
    relevant_data = []
    for table_info in tables:
        df = table_info['dataframe']
        # Convert DataFrame to string for simpler keyword matching across all cells
        df_string = df.to_string().lower()
        if any(keyword.lower() in df_string for keyword in keywords):
            relevant_data.append(table_info['markdown']) # Return markdown for LLM context
    return relevant_data

# --- Step 3: Query Optimization & Advanced RAG (Reranking) ---

# 3.1 Query Optimization (Simple Rule-Based Example)
def optimize_query(query):
    financial_keywords = ["revenue", "income", "expenses", "profit", "loss", "headcount", "capital expenditures"]
    if any(keyword in query.lower() for keyword in financial_keywords) and "2024" not in query and "q1" not in query:
        return query + " for Meta's Q1 2024 financial report"
    return query

# 3.2 Reranking Model Loading
def load_reranking_model(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        reranker = SentenceTransformer(model_name)
        return reranker
    except Exception as e:
        print(f"Error loading reranking model: {e}")
        return None

# --- Main RAG Pipeline Flow (Illustrative) ---

def run_rag_pipeline(pdf_path, query, embedding_model, generator_llm, reranking_model=None, tables=None):
    # Step 1: Preprocessing & Chunking
    full_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(full_text)

    # Step 2: Prepare all contexts (text chunks + table representations)
    all_contexts_for_embedding = prepare_context_for_embedding(text_chunks, tables if tables else [])
    # Store original chunk content or table markdown along with embedded context
    context_contents = [c['content'] for c in all_contexts_for_embedding]
    context_embeddings = embedding_model.encode(context_contents, convert_to_tensor=True)

    # Step 3.1: Query Optimization
    optimized_query = optimize_query(query)
    query_embedding = embedding_model.encode(optimized_query, convert_to_tensor=True)

    # Retrieval (Initial Top-N)
    cosine_scores = util.cos_sim(query_embedding, context_embeddings)[0]
    top_n_indices = np.argsort(cosine_scores.cpu().numpy())[::-1][:10] # Get top 10 for reranking

    retrieved_chunks_for_reranking = [context_contents[i] for i in top_n_indices]

    # Step 3.2: Reranking
    final_retrieved_context = ""
    if reranking_model and retrieved_chunks_for_reranking:
        cross_inp = [[optimized_query, chunk] for chunk in retrieved_chunks_for_reranking]
        cross_scores = reranking_model.predict(cross_inp)
        top_k_reranked_indices = np.argsort(cross_scores)[::-1][:3] # Get top 3 after reranking
        final_retrieved_context = "\n".join([retrieved_chunks_for_reranking[i] for i in top_k_reranked_indices])
    else:
        # If no reranker, just take top 3 from initial retrieval
        final_retrieved_context = "\n".join([retrieved_chunks_for_reranking[i] for i in range(min(3, len(retrieved_chunks_for_reranking)))])

    # Also, perform direct table lookup if keywords are present in query
    keywords_in_query = [word for word in query.lower().split() if word in ["revenue", "income", "expenses", "2024", "2023", "q1"]]
    if keywords_in_query and tables:
        relevant_table_md = find_in_tables_by_keywords(tables, keywords_in_query)
        if relevant_table_md:
            final_retrieved_context += "\n\nStructured Data from Tables:\n" + "\n".join(relevant_table_md)

    # Generation
    prompt = f"Based on the following context:\n{final_retrieved_context}\n\nAnswer the query: {query}"
    # The max_new_tokens parameter needs to be chosen carefully based on expected answer length
    generated_answer = generator_llm(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)[0]['generated_text']

    # Post-process generated answer to remove prompt and ensure it only contains the answer
    answer_start_index = generated_answer.lower().find(query.lower())
    if answer_start_index != -1 and answer_start_index < len(prompt): # Basic check to see if the query is repeated
        generated_answer = generated_answer[len(prompt):].strip()
        # Clean up any residual prompt elements or model's self-introductions
        if generated_answer.lower().startswith("based on the provided context:"):
            generated_answer = generated_answer[len("based on the provided context:"):].strip()
        if generated_answer.lower().startswith("answer:"):
            generated_answer = generated_answer[len("answer:"):].strip()
        if generated_answer.lower().startswith(query.lower()): # If it repeats the query
            generated_answer = generated_answer[len(query):].strip()
    
    # Simple check to remove potential prefix if the model starts with the prompt
    if generated_answer.startswith(prompt):
        generated_answer = generated_answer[len(prompt):].strip()


    return generated_answer.strip()

# --- Example Usage ---
if __name__ == "__main__":
    pdf_file_path = 'Meta’s Q1 2024 Financial Report.pdf'

    print("Loading models (this may take a moment)...")
    embedding_model = load_embedding_model()
    generator_llm = load_generator_llm()
    reranking_model = load_reranking_model()
    
    if not all([embedding_model, generator_llm, reranking_model]):
        print("Failed to load all models. Exiting.")
    else:
        print("Models loaded successfully.")
        print("Extracting tables...")
        tables = extract_tables_from_pdf(pdf_file_path)
        print("Tables extracted.")

        queries = [
            "What was Meta's revenue in Q1 2024?", # Step 1 test query
            "What were the key financial highlights for Meta in Q1 2024?", # Step 1 test query
            "What was Meta's net income in Q1 2024 compared to Q1 2023?", # Step 2 test query
            "Summarize Meta's operating expenses in Q1 2024.", # Step 2 test query
            "What was Meta's headcount as of March 31, 2024?", # Step 3 test query
            "How much did Meta spend on capital expenditures in Q1 2024, and what is the outlook for the full year 2024?", # Step 3 test query
            "Describe the growth in Meta's Family of Apps' key metrics and explain the expected trend for Reality Labs' operating losses." # Step 3 test query
        ]

        print("\n--- Running RAG Pipeline for Test Queries ---")
        for i, q in enumerate(queries):
            print(f"\nQuery {i+1}: {q}")
            answer = run_rag_pipeline(pdf_file_path, q, embedding_model, generator_llm, reranking_model, tables)
            print(f"Answer {i+1}: {answer}")