from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
from app.services import cv_service

router = APIRouter(tags=["vision"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


async def _read_image_upload(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {image.content_type}")
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    return contents


@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(image: UploadFile = File(...)):
    """Module A3 - detect + recognize a customer's face and log the visit."""
    contents = await _read_image_upload(image)
    try:
        result = cv_service.recognize_face(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {e}")
    return result


@router.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product(image: UploadFile = File(...)):
    """Module A2 - classify a product image into a category."""
    contents = await _read_image_upload(image)
    try:
        result = cv_service.classify_product(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Product classification failed: {e}")
    return result
