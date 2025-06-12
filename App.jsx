import { useEffect, useState } from 'react';
import './App.css'; // Import the CSS file for styling - IMPORTANT: App.css must be in the same directory!

function App() {
  // State variables for backend status
  const [backendStatus, setBackendStatus] = useState('Checking...');
  const [backendDetails, setBackendDetails] = useState({});

  // State variables for document upload functionality
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [uploadedDocuments, setUploadedDocuments] = useState([]);

  // State variables for chat interface
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]); // Stores {type: 'user'/'ai', text: '...', sources: [...]}
  const [answering, setAnswering] = useState(false);
  const [chatError, setChatError] = useState('');


  // Effect hook to fetch backend status when the component mounts
  useEffect(() => {
    const fetchBackendStatus = async() => {
      try {
        const response = await fetch('http://localhost:8000/api/status');
        if (!response.ok){
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setBackendStatus(data.status);
        setBackendDetails(data);
      } catch (error) {
        console.error("Failed to fetch backend status", error);
        setBackendStatus(`Error: ${error.message}`);
      }
    };
    fetchBackendStatus();
  }, []); // Empty dependency array ensures this runs only once on mount

  // Handles file selection for upload
  const handleFileChange = (event) => {
    if (event.target.files.length > 0) {
      setSelectedFile(event.target.files[0]);
      setUploadMessage(''); // Clear any previous upload messages
    } else {
      setSelectedFile(null);
      setUploadMessage('');
    }
  };

  // Handles the document upload process
  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadMessage('Please select a file first.');
      return;
    }

    setUploading(true); // Set uploading state to true to disable buttons
    setUploadMessage('Uploading...'); // Provide feedback to the user

    const formData = new FormData();
    formData.append('file', selectedFile); // Append the selected file to form data

    try {
      const response = await fetch('http://localhost:8000/api/upload-document/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Upload failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      setUploadMessage(`Successfully uploaded ${result.filename}! Processed ${result.chunks_processed} chunks.`);

      // Add the uploaded document to the list of uploaded documents
      setUploadedDocuments(prevDocs => [...prevDocs, {
        name: result.filename,
        chunks: result.chunks_processed,
        length: result.text_length
      }]);

      setSelectedFile(null); // Clear the selected file after successful upload
      document.getElementById('fileInput').value = ''; // Clear the file input field

    } catch (error) {
      console.error("Upload error:", error);
      setUploadMessage(`Upload failed: ${error.message}`); // Display error message
    } finally {
      setUploading(false); // Reset uploading state
    }
  };

  // Handles changes in the question input field
  const handleQuestionChange = (event) => {
    setCurrentQuestion(event.target.value);
    setChatError(''); // Clear any previous chat errors
  };

  // Handles sending the question to the AI backend
  const handleAskQuestion = async () => {
    if (!currentQuestion.trim()) {
      setChatError('Please enter a question.');
      return;
    }

    setAnswering(true); // Indicate that AI is thinking
    setChatError(''); // Clear any previous chat errors

    // Add user's question to chat history
    setChatHistory(prevHistory => [...prevHistory, { type: 'user', text: currentQuestion }]);
    setCurrentQuestion(''); // Clear the input field immediately

    try {
      const response = await fetch('http://localhost:8000/api/ask-question/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: currentQuestion }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`AI response failed: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();

      // Add AI's answer and sources to chat history
      setChatHistory(prevHistory => [...prevHistory, {
        type: 'ai',
        text: result.answer,
        sources: result.sources || [] // Ensure sources is an array, even if empty
      }]);

    } catch (error) {
      console.error("AI question error:", error);
      setChatError(`AI response failed: ${error.message}`);
    } finally {
      setAnswering(false); // Reset answering state
    }
  };

  // Allows sending the question by pressing Enter key
  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !answering) {
      event.preventDefault(); // Prevent default newline behavior in textarea
      handleAskQuestion();
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Dellreno CoPilot</h1>
      </header>

      <main className="app-main">
        {/* Document Management Panel */}
        <div className="document-management-panel">
          <h2>Document Management</h2>
          <p>Upload your technical manuals here.</p>

          <div className="upload-section">
            <input
              type="file"
              id="fileInput"
              onChange={handleFileChange}
              disabled={uploading}
              accept=".pdf,.docx,.txt" // Specify accepted file types
            />
            <button onClick={handleUpload} disabled={!selectedFile || uploading}>
              {uploading ? 'Uploading...' : 'Upload Document'}
            </button>
            {uploadMessage && <p className={`upload-message ${uploadMessage.startsWith('Upload failed') ? 'error' : 'success'}`}>{uploadMessage}</p>}
          </div>

          <h3>Uploaded Documents:</h3>
          {uploadedDocuments.length === 0 ? (
            <p>No documents uploaded yet.</p>
          ) : (
            <ul className="uploaded-docs-list">
              {uploadedDocuments.map((doc, index) => (
                <li key={index}>
                  <strong>{doc.name}</strong> ({doc.chunks} chunks, {doc.length} chars)
                </li>
              ))}
            </ul>
          )}

          <p>Backend Status: <strong>{backendStatus}</strong></p>
          <pre style={{ fontSize: '0.8em', color: '#666', whiteSpace: 'pre-wrap' }}>
            {/* Display detailed backend status in a pre-formatted block */}
            {JSON.stringify(backendDetails, null, 2)}
          </pre>
        </div>

        {/* AI Assistant Chat Interface Panel */}
        <div className="chat-interface-panel">
          <h2>AI Assistant Chat</h2>

          <div className="chat-history">
            {chatHistory.length === 0 ? (
              <p className="chat-welcome">Waiting for your questions...</p>
            ) : (
              // Map through chat history to display messages
              chatHistory.map((message, index) => (
                <div key={index} className={`chat-message ${message.type}`}>
                  <p className="message-text">
                    {/* Prefix for user and AI messages */}
                    {message.type === 'user' ? 'You: ' : 'AI: '}

                    {/* Conditional rendering for AI's answer to handle steps/paragraphs */}
                    {message.type === 'ai' && message.text.split('\n').map((line, lineIndex) => {
                      // Check if the line starts with a number followed by a period and space
                      const isStep = /^\d+\.\s/.test(line.trim());
                      // Render as <li> if it's a step, otherwise as <p>
                      return isStep ? <li key={lineIndex}>{line.trim()}</li> : <p key={lineIndex}>{line.trim()}</p>;
                    })}
                    {/* For user messages, just display the text directly */}
                    {message.type === 'user' && message.text}
                  </p>
                  {/* Display sources for AI messages */}
                  {message.type === 'ai' && message.sources && message.sources.length > 0 && (
                    <div className="message-sources">
                      <strong>Sources:</strong>
                      <ul>
                        {/* Render sources directly as they are now pre-formatted strings */}
                        {message.sources.map((source, srcIndex) => (
                          <li key={srcIndex}>{source}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))
            )}
            {answering && <p className="chat-loading">AI is thinking...</p>}
            {chatError && <p className="chat-error">{chatError}</p>}
          </div>

          {/* Chat Input Area */}
          <div className="chat-input-area">
            <textarea
              value={currentQuestion}
              onChange={handleQuestionChange}
              onKeyPress={handleKeyPress}
              placeholder="Ask your question about the documents..."
              rows="3"
              disabled={answering}
            />
            <button onClick={handleAskQuestion} disabled={answering || !currentQuestion.trim()}>
              Send
            </button>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>&copy; 2025 Dellreno. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;