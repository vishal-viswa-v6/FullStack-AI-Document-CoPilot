# FullStack-AI-Document-CoPilot
## Intelligent Document Analysis: A System using natural language, local LLM and Google API.

### **Features**

* **Document Upload**: Easily upload PDF, DOCX, and TXT documents.
* **Intelligent Q&A**: Ask questions in natural language and get answers extracted directly from your uploaded documents.
* **Contextual Sourcing**: AI responses are accompanied by clear citations indicating the source document and page number.
* **Scalable Architecture**: Built with FastAPI for the backend, React for the frontend, ChromaDB for vector storage, and Sentence Transformers for embeddings.
* **Google Gemini API Integration**: Utilizes Google's powerful Gemini LLM for generative AI capabilities.

---

### **Technology Stack**

* **Backend**:
    * **Python 3.9+**
    * **FastAPI**: For building the robust and high-performance API.
    * **LangChain**: For document splitting and general RAG orchestration.
    * **Sentence Transformers**: For generating document embeddings (using `all-MiniLM-L6-v2`).
    * **ChromaDB**: A lightweight, in-memory vector database for storing and retrieving document chunks.
    * **Google Gemini API**: For powering the conversational AI and generating answers.
* **Frontend**:
    * **React.js**: A modern JavaScript library for building user interfaces.
    * **Vite**: A fast build tool for the frontend development server.
    * **HTML5 & CSS3**: For structuring and styling the web application.

---

### **Getting Started**

Follow these steps to set up and run `industrial-copilot` locally.

#### **Prerequisites**

* [Python 3.9+](https://www.python.org/downloads/)
* [Node.js (LTS version)](https://nodejs.org/en/download/) & npm/yarn
* [Git](https://git-scm.com/downloads)
* **Google API Key**: You'll need a Google API Key with access to the Gemini API. [Get your API Key here](https://aistudio.google.com/app/apikey).

# **1. Clone the Repository**

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Create a .env file and add your Google API Key
# (replace YOUR_GOOGLE_API_KEY_HERE with your actual key)
echo "GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY_HERE" > .env

# Run the backend server
uvicorn main:app --reload --port 8000

# Install dependencies
npm install # or yarn install

# Run the frontend development server
npm run dev # or yarn dev
