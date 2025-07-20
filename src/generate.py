import os
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import HumanMessage, SystemMessage
# Note: ChatGoogleGenerativeAI is imported inside function for testability

def generate_response(prompt: str) -> str:
    """
    Generate a response using Gemini model with a financial domain system prompt.
    """
    try:
        
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-flash",
            temperature=0.3,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        messages = [
            SystemMessage(content="Tu es un conseiller financier expert. Réponds de manière claire, précise et conforme à la réglementation."),
            HumanMessage(content=prompt)
        ]

        result = llm.invoke(messages)
        return result.content
    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        return "[ERROR] Impossible de générer une réponse. Veuillez réessayer."
