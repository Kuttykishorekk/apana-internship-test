import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langsmith import traceable

# Initialize LLM with optimized settings
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.3,  # Lower temperature for more consistent responses
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# System prompt for financial expertise
SYSTEM_PROMPT = """Tu es un expert en finance personnelle et produits financiers français. 
Tu dois répondre de manière précise, concise et factuelle aux questions sur les produits financiers français.
Utilise tes connaissances à jour sur la réglementation et les taux en vigueur.
Donne des réponses claires et utiles pour aider les utilisateurs à comprendre les concepts financiers."""

@traceable(name="Generate Response (Gemini)")
def generate_response(prompt: str) -> str:
    """
    Generate a response using Gemini with financial domain expertise.
    
    Args:
        prompt (str): The user's financial question
        
    Returns:
        str: Generated response or error message
    """
    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"[GENERATION ERROR] {e}")
        return "[ERROR] Impossible de générer une réponse. Veuillez réessayer."