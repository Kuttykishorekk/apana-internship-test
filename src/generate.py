import os
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import HumanMessage

def generate_response(prompt: str) -> str:
    """
    Generate a response using Gemini model from LangChain.
    """
    try:
        # ✅ move import inside to allow monkeypatching in tests
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        result = llm.invoke([HumanMessage(content=prompt)])
        return result.content
    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        return "[ERROR] Impossible de générer une réponse. Veuillez réessayer."
