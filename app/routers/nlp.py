from fastapi import APIRouter, HTTPException

from app.schemas import SentimentRequest, SentimentResponse
from app.services import nlp_service

router = APIRouter(tags=["nlp"])


@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    """Module B2 - classify review/feedback text as Positive/Negative/Neutral."""
    try:
        result = nlp_service.analyze_sentiment(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")
    return result
