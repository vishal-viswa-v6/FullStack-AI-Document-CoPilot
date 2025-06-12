import re # Import regex for advanced parsing
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import fitz
from docx import Document
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai # Corrected import for Google Gemini


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
# Debug print is optional, can be removed once confident in .env loading
print(f"DEBUG: OPENROUTER_API_KEY loaded: {os.getenv('OPENROUTER_API_KEY')}")
app = FastAPI()

# --- ChromaDB Setup ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    documents_collection = chroma_client.get_or_create_collection(name="industrial_docs")
    logger.info("ChromaDB Collection 'industrial_docs' Initialized")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB collection: {e}") # Corrected colon
    documents_collection = None

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:3000",
    # Add other frontend URLs here when deployed, e.g., "https://yourfrontenddomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Google Gemini Generative LLM Configuration (Replaced OpenRouter) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    logger.info("Google Gemini API configured successfully.")
else:
    logger.error("GOOGLE_API_KEY not found in environment variables. Gemini API will not function.")

GEMINI_LLM_MODEL="gemma-3-27b-it"

# --- Text Extraction Utility Functions ---a
def extract_text_from_pdf(file_content: bytes) -> list[dict]:
    """Extracts text from PDF FILE"""
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        pages_data = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num) # Load page here
            text = page.get_text()
            if text.strip():
                pages_data.append({"text": text, "page_number": page_num + 1})
        return pages_data
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail=f"PDF EXTRACTION FAILED: {e}") # Corrected EXTRACTION

def extract_from_docx(file_content: bytes) -> str:
    """Extracts text from a DOCX file."""
    try:
        # 'import io' is now at the top, so no need to import it here again
        doc = Document(io.BytesIO(file_content))
        full_text = ""
        for paragraph in doc.paragraphs:
            full_text += paragraph.text + "\n"
        return [{"text": full_text, "page_number": 1}] if full_text.strip() else []
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        raise HTTPException(status_code=500, detail=f"DOCX EXTRACTION FAILED: {e}")

def extract_text_from_txt(file_content: bytes) -> list[dict]:
    """Extracts text from a plain text file."""
    try:
        full_text = file_content.decode('utf-8')
        return [{"text": full_text, "page_number": 1}] if full_text.strip() else []
    except Exception as e:
        logger.error(f"Error decoding text file: {e}")
        raise HTTPException(status_code=500, detail=f"TXT FILE DECODING FAILED: {e}")

# --- Local Embedding Model Configuration ---
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("Local embedding model 'all-MiniLM-L6-v2' loaded successfully")
except Exception as e:
    logger.error(f"Failed to load local embedding model: {e}")
    embedding_model = None

# --- Utility Function for Embedding ---
def get_embedding(text: str) -> list[float]:
    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Embedding Model not loaded. Check server logs")
    if not text.strip():
        return []

    try:
        embedding = embedding_model.encode(text).tolist()
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding with local model: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}. Check text content or model integrity.")

# --- Document Processing (Chunking and Embedding) ---
def process_document_for_rag(document_name: str, extracted_raw_data: list[dict]):
    logger.info(f"Starting RAG Processing for document: {document_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False,
    )

    # These lists are to collect data for a *single batch add* to ChromaDB
    chunk_ids = []
    chunk_embeddings = []
    chunk_texts = []
    chunk_metadatas = [] # Corrected variable name consistency

    processed_chunks_info_for_api_response = []

    # This list is just for the API response summary (Moved outside loop)

    for page_data in extracted_raw_data:
        page_text = page_data["text"]
        page_number = page_data["page_number"]

        # Ensure create_documents handles a single string as input
        chunks_from_page = text_splitter.create_documents([page_text]) # Wrap in a list

        if not chunks_from_page:
            logger.warning(f"No chunks generated from page {page_number} of {document_name}. Skipping.")
            continue

        for i, chunk in enumerate(chunks_from_page): # Loop through chunks from THIS page
            chunk_text = chunk.page_content
            embedding = get_embedding(chunk_text)

            if embedding_model is None:
                logger.error("Embedding model not loaded, cannot process chunks for database.")
                raise HTTPException(status_code=500, detail="Embedding model not loaded.")
            if not embedding:
                logger.warning(f"Empty embedding for chunk {i+1} on page {page_number} of {document_name}. Skipping.")
                continue

            # Create a unique ID that includes page number for robustness
            unique_id = f"{document_name}_page_{page_number}_chunk_{i+1}"

            chunk_ids.append(unique_id)
            chunk_embeddings.append(embedding)
            chunk_texts.append(chunk_text)
            chunk_metadatas.append({
                "document_name": document_name,
                "chunk_number": i + 1,
                "page_number": page_number, # ADDED PAGE NUMBER METADATA
            })

            processed_chunks_info_for_api_response.append({
                "chunk_number": i + 1,
                "text_preview": chunk_text[:200] + "...",
                "embedding_length": len(embedding),
                "document_name": document_name,
                "page_number": page_number, # ADDED PAGE NUMBER TO RESPONSE SUMMARY
            })

            logger.info(f"Processed chunk {i+1}/{len(chunks_from_page)} (page {page_number}) for {document_name}, embedding length: {len(embedding)}")

    # --- END OF LOOP ---

    # --- ChromaDB ADDITION (MOVED OUTSIDE THE LOOP) ---
    if documents_collection is not None and chunk_ids:
        try:
            documents_collection.add(
                ids=chunk_ids,
                embeddings=chunk_embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            logger.info(f"Added {len(chunk_ids)} chunks from {document_name} to ChromaDB.")
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to store document in database: {e}")
    elif documents_collection is None: # Handle case where collection itself failed to initialize
        logger.error("ChromaDB collection not initialized, cannot store chunks.")
        raise HTTPException(status_code=500, detail="Document storage failed: Database not ready.")

    # --- FINAL LOG AND RETURN (MOVED OUTSIDE THE LOOP) ---
    logger.info(f"Finished RAG processing for {document_name}. Total chunks: {len(processed_chunks_info_for_api_response)}")
    return processed_chunks_info_for_api_response

# --- Semantic Search Utility function (Task 2.4, Placed BEFORE ask_question) ---
def retrieve_relevant_chunks(query_text: str, n_results: int=5) -> list[dict]: # Corrected function name
    """
    Retrieves the most semantically relevant chunks from ChromaDB for a given query.
    """
    if embedding_model is None:
        logger.error("Embedding model not loaded, cannot perform retrieval.") # Corrected typo
        raise HTTPException(status_code=500, detail="Embedding model not loaded")
    if documents_collection is None:
        logger.error("ChromaDB collection not initialized, cannot perform retrieval.") # Corrected typo
        raise HTTPException(status_code=500, detail="Database not ready.")
    query_embedding = get_embedding(query_text)

    logger.info(f"DEBUG_RETRIEVAL: Query text: '{query_text[:50]}'")
    logger.info(f"DEBUG_RETRIEVAL: Query embedding length: {len(query_embedding)}")
    logger.info(f"DEBUG_RETRIEVAL: n_results: {n_results}")

    try:
        results = documents_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results, # Corrected argument
            include=['documents','metadatas','distances']
        )
        relevant_chunks_info = [] # Corrected variable name
        if results['documents'] and results ['documents'][0]:
            for i in range (len(results['documents'][0])):
                chunk_text = results['documents'][0][i]
                metadata = results['metadatas'][0][i] # Corrected key
                distance = results['distances'][0][i] # Corrected key

                relevant_chunks_info.append({
                    "text": chunk_text,
                    "document_name": metadata.get("document_name", "N/A"),
                    "chunk_number": metadata.get("chunk_number", "N/A"),
                    "page_number": metadata.get("page_number", "N/A"),
                })
        logger.info(f"Retrieved {len(relevant_chunks_info)} relevant chunks for query: '{query_text[:50]}...'") # Corrected typo and variable name
        return relevant_chunks_info

    except Exception as e:
        logger.error(f"Error retrieving chunks from ChromaDB: {e}") # Corrected typo
        raise HTTPException(status_code=500, detail=f"Failed to retrieve relevant documents: {e}") # Corrected typo

# --- FastAPI Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "IndustrialCoPilot is running!"}

@app.get("/api/status")
async def get_status():
    openrouter_api_key_status = "set" if os.getenv("OPENROUTER_API_KEY") else "not set"
    hf_api_token_status = "set" if os.getenv("HF_API_TOKEN") else "not set"
    local_embedding_model_status = "loaded" if embedding_model is not None else "failed to load"
    return {
        "status": "ok",
        "OPENROUTER_API_KEY": openrouter_api_key_status,
        "HF_API_TOKEN": hf_api_token_status,
        "LOCAL_EMBEDDING_MODEL": local_embedding_model_status
    }

@app.post("/api/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    logger.info(f"Received file upload: {file.filename}, Content-Type: {file.content_type}")
    file_content = await file.read()

    if file.content_type == "application/pdf":
        extracted_raw_data = extract_text_from_pdf(file_content) # Changed variable name
    elif file.content_type == "application/vnd.openxmlformats-officedocuments.wordprocessingml.document":
        extracted_raw_data = extract_from_docx(file_content) # Changed variable name
    elif file.content_type == "text/plain":
        extracted_raw_data = extract_text_from_txt(file_content) # Changed variable name


    total_chars = sum(len(page_data["text"]) for page_data in extracted_raw_data) if extracted_raw_data else 0
    logger.info(f"Successfully extracted text from {file.filename}. Text length: {total_chars}")
    processed_chunks_summary = process_document_for_rag(file.filename, extracted_raw_data)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "extracted_text_preview": processed_chunks_summary[0]['text_preview'] if processed_chunks_summary else "", # Use preview from first chunk
        "text_length": total_chars,
        "chunks_processed": len(processed_chunks_summary),
        "first_chunk_preview": processed_chunks_summary[0]['text_preview'] if processed_chunks_summary else None,
        "first_chunk_embedding_length": processed_chunks_summary[0]['embedding_length']if processed_chunks_summary else None

    }

# --- NEW Endpoint for Asking Questions (Task 2.4 - Now using Google Gemini) ---
@app.post("/api/ask-question/")
async def ask_question(query: dict):
    user_question = query.get("question")
    if not user_question:
        raise HTTPException(status_code=400, detail="Question field is required.")

    logger.info(f"Received question: '{user_question}'")

    relevant_chunks = retrieve_relevant_chunks(user_question, n_results=3)

    if not relevant_chunks:
        logger.warning(f"No relevant chunks found for query: '{user_question}'")
        return {"answer": "I couldn't find any relevant information in the uploaded documents to answer your question.", "sources": []}

    context_text = "\n".join([chunk["text"] for chunk in relevant_chunks])

    # Format sources exactly as requested
    formatted_sources = []
    unique_sources = set()
    for chunk in relevant_chunks:
        doc_name_clean = chunk["document_name"].replace(".pdf", "").replace(".docx", "").replace(".txt", "")
        source_string = f"{doc_name_clean} || Page: {chunk['page_number']}"
        unique_sources.add(source_string)

    # Construct the prompt for the generative LLM (MODIFIED for Gemini)
    gemini_prompt = (
        "You are an expert industrial operations assistant. "
        "Your goal is to provide precise, accurate, and concise answers based ONLY on the provided context from industrial documents. "
        "If the user asks for a procedure or steps, provide the full, step-by-step procedure/steps as a numbered list. "
        "Each step should be on a new line, prefixed by its number and a period (e.g., '1. First step', '2. Second step'). "
        "If the answer is a paragraph, provide it as a single paragraph. "
        "DO NOT include any citations or source references in your answer text itself. "
        "If the answer cannot be found in the context, clearly state that you don't have enough information from the documents. "
        "DO NOT invent information or stray from the given context. \n\n"
        f"Context:\n{context_text}\n\nQuestion: {user_question}"
    )

    try:
        if not GOOGLE_API_KEY: # Check directly here
            raise HTTPException(status_code=500, detail="Google API key is not set.")

        # Make the API call to Google Gemini
        model_client = genai.GenerativeModel(GEMINI_LLM_MODEL)
        response = model_client.generate_content(gemini_prompt)

        # Gemini's response structure is slightly different
        llm_answer = response.text
        logger.info(f"LLM generated answer for query: '{user_question}'")
        return {"answer": llm_answer, "sources": formatted_sources}

    except Exception as e:
        logger.error(f"Error calling Google Gemini LLM: {e}")
        error_detail = f"Failed to generate answer: {e}"
        if 'response' in locals() and hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
             error_detail = f"Blocked by safety settings: {response.prompt_feedback.block_reason.name}"
        elif 'response' in locals() and hasattr(response, 'candidates') and not response.candidates:
             error_detail = "No candidates generated (possibly blocked by safety settings or content policy)."

        raise HTTPException(status_code=500, detail=error_detail)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)