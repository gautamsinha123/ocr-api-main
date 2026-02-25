from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import tempfile, fitz

from app.services.paddleocr_service import paddle_ocr_and_annotate
from app.utils.regex_utils import extract_document_ids

router = APIRouter(prefix="/paddleocr", tags=["PaddleOCR"])

@router.post("/pdf")
async def paddleocr_pdf(file: UploadFile = File(...)):
    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "Only PDF files allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=120)

    img_path = pdf_path + "_page1.png"
    pix.save(img_path)
    doc.close()

    result = paddle_ocr_and_annotate(img_path)
    document_ids = extract_document_ids(result["raw_text"])

    return {
        "filename": file.filename,
        "pages": 1,
        "texts": result["texts"],
        "raw_text": result["raw_text"],
        "document_ids": document_ids,
        "document_type": list(document_ids.keys())[0] if document_ids else None,
        "execution_time": result["execution_time"]
    }
