Customer Success Librarian
An AI-powered web application that retrieves and presents ManageEngine customer success stories from a Firestore database.
It features a modern, card-based UI and can be deployed to Google Cloud Run for public access.

🚀 Features
Conversational Search:
Users can query in natural language for customer success stories.

Google Cloud AI Integration:
Uses Google Vertex AI for LLM-powered query understanding and response generation.

Firestore Database:
Stores and retrieves customer success stories with metadata.

Modern UI:
Clean, responsive, full-screen interface inspired by ChatGPT’s style.

Cloud Deployable:
Ready for Google Cloud Run deployment with environment variables.

chatbot-api/
│
├── main.py          # FastAPI backend, Firestore & Vertex AI logic
├── requirements.txt # Python dependencies
├── frontend/
│   ├── index.html   # Main UI
│   ├── style.css    # Styling
│   └── script.js    # Client-side logic
└── README.md

🛠️ Tech Stack
Backend:

 - Python 3.12
 - FastAPI
 - Google Cloud Firestore
 - Google Vertex AI
 - Uvicorn
   
Frontend:

 - HTML5, CSS3, JavaScript
 - Responsive flexbox & card-based layout

Deployment:

 - Google Cloud Run
 - Artifact Registry
 - Buildpacks/Docker

🧠 How It Works
User Query:
The frontend sends the query to the FastAPI backend (/api/query).

LLM Classification:
Vertex AI classifies if the query is relevant.

Firestore Search:
If relevant, the backend retrieves matching customer stories from Firestore.

Response Formatting:
Data is sent back with fields like logo, name, designation, LinkedIn, etc.

Frontend Display:
Results are rendered as responsive cards.
