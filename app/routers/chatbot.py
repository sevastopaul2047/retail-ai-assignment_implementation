from fastapi import APIRouter, HTTPException

from app.schemas import ChatbotRequest, ChatbotResponse
from app.services import chatbot_service

router = APIRouter(tags=["chatbot"])


@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_reply(payload: ChatbotRequest):
    """Module B3 - hybrid rule-based + ML FAQ chatbot."""
    try:
        result = chatbot_service.get_bot_reply(payload.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot failed: {e}")
    return result
