from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.api.paddleocr_image import router as paddleocr_image_router
from app.api.paddleocr_pdf import router as paddleocr_pdf_router

app = FastAPI(
    title="FastAPI OCR Service",
    root_path="/ocr-api",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(paddleocr_image_router)
app.include_router(paddleocr_pdf_router)

@app.get("/")
def root():
    return {"message": "Welcome to FastAPI PaddleOCR APIs"}
