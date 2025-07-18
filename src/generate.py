import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

def generate_response(prompt: str) -> str:
    """
    Generate a response using Gemini model from LangChain.
    """
    try:
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