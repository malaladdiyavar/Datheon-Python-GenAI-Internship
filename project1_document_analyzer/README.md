# AI Document Analyzer

## Overview
The AI Document Analyzer is a Python-based Generative AI application developed for the Datheon Python + Generative AI & LangChain Internship. It extracts structured information from invoice and resume text using Google's Gemini API.

## Features
- Invoice and resume text analysis
- Prompt engineering with explicit extraction rules
- Structured JSON generation
- Pydantic schema validation
- Missing-field identification
- JSON output saved to a file
- Secure API key configuration using environment variables

## Technologies Used
- Python
- Google Gemini API
- Google Gen AI Python SDK
- Pydantic
- python-dotenv
- JSON

## Project Structure
```text
project1_document_analyzer/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

The `.env` file contains the API key and must not be uploaded to GitHub.

## Installation
1. Install Python.
2. Create and activate a virtual environment.
3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Create a `.env` file containing:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your own Gemini API key.

## Run the Application

```bash
python main.py
```

Choose invoice or resume, paste the document text, and type `END` on a new line.

The application prints validated JSON and saves the result in a JSON file.

## Limitations
- The application currently accepts pasted text rather than directly reading PDF or Word files.
- Extraction quality depends on the supplied text and the AI model.
- Generated results should be checked for accuracy before practical use.

## Internship
Developed as a project for the Python + Generative AI & LangChain Internship.
