"""
Pydantic models for request/response validation across all endpoints.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ---------- Vision ----------
class ProductClassificationResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict[str, float]


class FaceRecognitionResponse(BaseModel):
    status: str = Field(..., description="returning_customer | unknown_customer | no_face_detected")
    customer: Optional[str] = None
    confidence: Optional[float] = None
    face_box: Optional[list[int]] = None


# ---------- NLP ----------
class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["The shoes I ordered are amazing!"])


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: Optional[float] = None
    cleaned_text: str


# ---------- Chatbot ----------
class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, examples=["What is your return policy?"])


class ChatbotResponse(BaseModel):
    reply: str
    intent: str
    method: str = Field(..., description="rule_based | ml_fallback")


# ---------- Dashboard ----------
class DashboardStatsResponse(BaseModel):
    visits: dict
    sentiment: dict
