\# RAG Document Question Answering System



\## Project Overview

This project is a Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from local documents. It uses TF-IDF and cosine similarity to find relevant text and Google Gemini to generate answers based on that text.



If Gemini is temporarily unavailable, the application provides a local fallback answer using relevant sentences from the retrieved documents.



\## Features

\- Reads TXT and PDF documents.

\- Splits documents into searchable text chunks.

\- Retrieves relevant information using TF-IDF and cosine similarity.

\- Generates document-grounded answers using Google Gemini.

\- Displays source filenames.

\- Retries temporary Gemini server errors.

\- Provides a local fallback when Gemini is unavailable.

\- Includes an interactive command-line menu.



\## Technologies Used

\- Python

\- Google Gemini API

\- TF-IDF

\- Cosine similarity

\- Scikit-learn

\- PyPDF

\- python-dotenv



\## Project Structure

```text

project2\_rag\_document\_qa/

├── documents/

│   └── project\_notes.txt

├── main.py

├── requirements.txt

├── README.md

└── .env

```



\## Setup Instructions



1\. Install Python.

2\. Open a terminal in the project directory.

3\. Create and activate a virtual environment if needed.

4\. Install dependencies:



&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```



5\. Create a `.env` file in the project directory:



&#x20;  ```text

&#x20;  GEMINI\_API\_KEY=your\_gemini\_api\_key\_here

&#x20;  ```



6\. Add your TXT or PDF files to the `documents` folder.

7\. Run the application:



&#x20;  ```bash

&#x20;  python main.py

&#x20;  ```



\## Usage

1\. Select \*\*1\*\* to ask a question.

2\. Enter a question about the documents.

3\. Read the answer and source filename.

4\. Select \*\*2\*\* to rebuild the document index after changing documents.

5\. Select \*\*3\*\* to exit.



\## How It Works

1\. The application loads TXT and PDF files.

2\. It splits document text into chunks.

3\. TF-IDF converts the chunks into numerical representations.

4\. Cosine similarity ranks chunks by relevance to the user's question.

5\. Gemini receives the question and retrieved context to generate an answer.

6\. If Gemini is unavailable, a local fallback selects relevant document sentences.



\## Limitations

\- Answers depend on the content and quality of the documents.

\- Scanned PDFs may require OCR before their text can be extracted.

\- Gemini availability depends on the API service and quota.

\- The local fallback matches sentences using text similarity and may not produce a complete answer.



\## Security

Never commit your actual `.env` file or API key to GitHub. Keep secrets private.



\## Conclusion

This project demonstrates document retrieval, prompt engineering, LLM integration, source tracking, and fallback handling in a Python-based RAG application.



