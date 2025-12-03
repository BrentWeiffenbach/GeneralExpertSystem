
# Troubleshooting System (MVP)

This project demonstrates AI orchestration and retrieval-augmented generation (RAG) to power the Troubleshooting System. It ships with:

- **Admin / Troubleshooting intake** page for document uploads, semantic chunking, and conversational knowledge capture.
- **ChatGPT-style user assistant** grounded in the curated vector database.
- **Debug view** to inspect the live chunk store.

## Prerequisites

- Python 3.10+
- A Google Gemini API key exported as `GEMINI_API_KEY` (or stored in a `.env` file)
- `pip` + `venv`

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer, run `./setup.sh` to automate these steps. When prompted, paste your Gemini API key.

## Running the MVP

```bash
. .venv/bin/activate
python app.py
```

Then open <http://127.0.0.1:5000>.

### Views

- **User Assistant** – ask troubleshooting questions. The backend searches the vector store, feeds the context to Gemini, and auto-learns new insights when confidence is high.
- **Admin / Expert** – upload PDFs or CSVs, see chunk annotations, and collaborate with an “intake AI” that decides when a conversation is ready to archive. You can also force-save anything valuable.
- **Chunk Debug** – review everything inside the SQLite-backed vector database.

### Data flow

1. Documents are parsed, chunked, and summarized with Gemini before entering the vector store.
2. Both expert and user chats run through Gemini with RAG context.
3. A secondary Gemini call determines whether a fresh chunk should be added based on the ongoing conversation.

## Testing

```bash
. .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_vector_store.py
```

## Project structure

```text
app.py                  # Flask app + REST endpoints
gemini.py               # Gemini helper wrapper
ges_core/
	document_processor.py # ingestion + chunk metadata + conversation summaries
	rag.py                # retrieval augmented generation engine
	vector_store.py       # SQLite vector store helper
templates/              # Jinja templates for all views
uploads/, data/         # runtime dirs for files + SQLite DB
tests/                  # pytest coverage
```