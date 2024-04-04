import React, { useState } from "react";
import axios from "axios";
import "./App.css"; // Import CSS file for styling

function App() {
  const [inputText, setInputText] = useState("");
  const [responseChunks, setResponseChunks] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const { data } = await axios.post("http://127.0.0.1:5000/chatbot", {
        input: inputText,
      });
      setResponseChunks(data.response_chunks);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>DevSavvy Chatbot: Unlocking Coding Wisdom for You</h1>
      </header>
      <main className="main">
        <form onSubmit={handleSubmit} className="form">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Share Your Code Dreams..."
            className="input"
          />
          <button type="submit" className="button">
            Send
          </button>
        </form>
        <div className="response-container">
          {responseChunks.map((chunk, index) => (
            <p key={index}>{chunk}</p>
          ))}
        </div>
      </main>
    </div>
  );
}

export default App;
