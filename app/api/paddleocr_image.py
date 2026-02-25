from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile, shutil

from app.services.paddleocr_service import paddle_ocr_and_annotate
from app.utils.regex_utils import extract_document_ids

router = APIRouter(prefix="/paddleocr", tags=["PaddleOCR"])

@router.post("/predict")
async def paddleocr_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
        raise HTTPException(400, "Only image files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        img_path = tmp.name

    result = paddle_ocr_and_annotate(img_path)
    document_ids = extract_document_ids(result["raw_text"])

    return {
        "filename": file.filename,
        "texts": result["texts"],
        "raw_text": result["raw_text"],
        "document_ids": document_ids,
        "document_type": list(document_ids.keys())[0] if document_ids else None,
        "execution_time": result["execution_time"]
    }
