
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None
MODEL_NAME = "gemini-3.8-flash"

chunks = []
sources = []
vectorizer = None
document_vectors = None


# ==================================================
# 2. SPLIT TEXT INTO CHUNKS
# ==================================================

def split_text(text, chunk_size=150, overlap=30):
    words = text.split()
    result = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])

        if chunk.strip():
            result.append(chunk)

        if start + chunk_size >= len(words):
            break

    return result


# ==================================================
# 3. LOAD TXT AND PDF DOCUMENTS
# ==================================================

def load_documents():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    for file_path in DOCUMENTS_DIR.iterdir():
        if not file_path.is_file():
            continue

        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8")

            elif file_path.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )

            else:
                continue

            if text.strip():
                documents.append((file_path.name, text))

        except Exception as error:
            print(f"Could not read {file_path.name}: {error}")

    return documents


# ==================================================
# 4. BUILD SEARCH INDEX
# ==================================================

def build_index():
    global chunks, sources, vectorizer, document_vectors

    documents = load_documents()
    chunks = []
    sources = []

    for filename, text in documents:
        for chunk in split_text(text):
            chunks.append(chunk)
            sources.append(filename)

    if not chunks:
        vectorizer = None
        document_vectors = None
        print("No readable TXT or PDF documents found.")
        return False

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    document_vectors = vectorizer.fit_transform(chunks)

    print(f"\nIndexed {len(documents)} document(s).")
    print(f"Created {len(chunks)} searchable chunk(s).")
    return True


# ==================================================
# 5. RETRIEVE RELEVANT CHUNKS
# ==================================================

def retrieve_context(question, top_k=3):
    if vectorizer is None or document_vectors is None:
        return []

    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(
        question_vector, document_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        results.append({
            "text": chunks[index],
            "source": sources[index],
            "score": float(scores[index])
        })

        if len(results) >= top_k:
            break

    return results


# ==================================================
# 6. LOCAL FALLBACK ANSWER
# ==================================================

def local_fallback_answer(question, results):
    """Find relevant sentences locally without calling Gemini."""

    sentences = []

    for item in results:
        parts = item["text"].replace("\n", " ").split(".")
        for part in parts:
            sentence = part.strip()
            if sentence:
                sentences.append((sentence, item["source"]))

    if not sentences:
        print("No sentences available in the retrieved documents.")
        return

    sentence_texts = [item[0] for item in sentences]

    local_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        matrix = local_vectorizer.fit_transform(
            [question] + sentence_texts
        )
        scores = cosine_similarity(
            matrix[0:1], matrix[1:]
        ).flatten()
    except ValueError:
        print("Could not find enough matching text.")
        return

    ranked = scores.argsort()[::-1]
    selected = []

    for index in ranked:
        if scores[index] <= 0:
            continue

        sentence, source = sentences[index]

        if sentence not in [item[0] for item in selected]:
            selected.append((sentence, source))

        if len(selected) == 3:
            break

    print("\n" + "=" * 50)
    print("LOCAL DOCUMENT ANSWER (Gemini unavailable)")
    print("=" * 50)

    if selected:
        for sentence, source in selected:
            print(f"- {sentence}.")
    else:
        print(
            "No sufficiently relevant sentences were found. "
            "Try rephrasing your question."
        )

    print("\nSources:")
    for source in dict.fromkeys(item[1] for item in selected):
        print(f"- {source}")

    print("=" * 50)


# ==================================================
# 7. ANSWER USING GEMINI, WITH LOCAL FALLBACK
# ==================================================

def answer_question(question):
    print("\nSearching your documents...")
    results = retrieve_context(question)

    if not results:
        print("No relevant information was found in your documents.")
        return

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}"
        for item in results
    )

    prompt = f"""
Answer the question using only the document context below.
Do not invent facts. If the answer is missing, say so clearly.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

Give a clear, direct answer.
"""

    if client is None:
        print("Gemini API key is missing; using local search instead.")
        local_fallback_answer(question, results)
        return

    for attempt in range(3):
        try:
            print(f"Sending question to Gemini ({attempt + 1}/3)...")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            answer = response.text

            if not answer or not answer.strip():
                raise ValueError("Gemini returned an empty answer.")

            print("\n" + "=" * 50)
            print("GEMINI ANSWER")
            print("=" * 50)
            print(answer.strip())

            print("\nSources:")
            for source in dict.fromkeys(
                item["source"] for item in results
            ):
                print(f"- {source}")

            print("=" * 50)
            return

        except ServerError as error:
            error_text = str(error)

            if "503" in error_text and attempt < 2:
                delay = 3 * (2 ** attempt)
                print(f"Gemini is busy. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("\nGemini is unavailable. Using local fallback.")
                print("Details:", repr(error))
                local_fallback_answer(question, results)
                return

        except Exception as error:
            print("\nGemini request failed. Using local fallback.")
            print("Details:", repr(error))
            local_fallback_answer(question, results)
            return


# ==================================================
# 8. MAIN MENU
# ==================================================

def main():
    print("\n" + "=" * 50)
    print("       RAG DOCUMENT Q&A ASSISTANT")
    print("=" * 50)

    if not build_index():
        return

    while True:
        print("\nMENU")
        print("1. Ask a question")
        print("2. Rebuild document index")
        print("3. Exit")

        try:
            choice = input("\nChoose an option (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram stopped.")
            break

        if choice == "1":
            question = input("\nEnter your question: ").strip()
            if question:
                answer_question(question)
            else:
                print("Please enter a question.")

        elif choice == "2":
            print("\nRebuilding index...")
            build_index()

        elif choice == "3":
            print("Thank you for using the RAG Assistant!")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
