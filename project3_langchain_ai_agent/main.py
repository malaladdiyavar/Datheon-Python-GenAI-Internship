
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load the API key from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )

# Configure the Gemini model through LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=API_KEY,
    temperature=0.3,
    max_retries=2,
)

SYSTEM_PROMPT = """
You are an AI Study Assistant for college students.

Your responsibilities:
- Explain AI, machine learning, Python, and computer science topics.
- Use simple language and examples.
- Give step-by-step explanations when requested.
- If you are unsure, say so instead of inventing facts.
- Keep answers organized and useful for learning.
"""

# Store the conversation for this running session
history = [SystemMessage(content=SYSTEM_PROMPT)]


def ask_question(question):
    """Send a question with the conversation history to Gemini."""
    history.append(HumanMessage(content=question))

    try:
        response = llm.invoke(history)
        answer = response.content

        # Handle text and other possible content blocks
        if isinstance(answer, list):
            answer = "\n".join(
                str(block.get("text", ""))
                if isinstance(block, dict)
                else str(block)
                for block in answer
            )

        answer = str(answer).strip()

        if not answer:
            answer = "The AI returned an empty response."

        history.append(AIMessage(content=answer))
        return answer

    except Exception as error:
        # Remove the failed question so it is not left in history
        history.pop()
        print("\nSorry, the AI request failed.")
        print("Check your internet connection, API key, quota, or model availability.")
        print("Error details:", str(error))
        return None


def main():
    print("=" * 55)
    print("             LANGCHAIN AI STUDY ASSISTANT")
    print("=" * 55)
    print("Ask questions about Python, AI, ML, and computer science.")
    print("Type 'clear' to clear conversation history.")
    print("Type 'exit' to quit.")
    print("=" * 55)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() == "exit":
            print("Goodbye! Keep learning.")
            break

        if question.lower() == "clear":
            history.clear()
            history.append(SystemMessage(content=SYSTEM_PROMPT))
            print("Conversation history cleared.")
            continue

        print("\nAssistant: Thinking...")

        answer = ask_question(question)

        if answer:
            print("\nAssistant:")
            print(answer)


if __name__ == "__main__":
    main()
