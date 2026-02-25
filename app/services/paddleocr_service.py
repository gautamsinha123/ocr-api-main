import os
import re
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Default models with all heavy processing disabled
ocr = PaddleOCR(
    lang="en",
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
)

# Patterns that indicate correct orientation (readable text)
_VALID_PATTERNS = [
    re.compile(r'\d{2}/\d{2}/\d{4}'),           # Date: DD/MM/YYYY
    re.compile(r'[A-Z]{5}\d{4}[A-Z]'),          # PAN format
    re.compile(r'\d{4}\s?\d{4}\s?\d{4}'),       # Aadhaar format
    re.compile(r'INCOME|TAX|DEPARTMENT|GOVT|INDIA|NAME|FATHER', re.I),
    re.compile(r'PERMANENT|ACCOUNT|NUMBER|CARD', re.I),
    re.compile(r'DRIVING|LICENCE|LICENSE|TRANSPORT', re.I),
    re.compile(r'PASSPORT|REPUBLIC', re.I),
]


def _resize(img: np.ndarray, max_dim: int = 1000) -> np.ndarray:
    """Downscale large images. Phone photos are way too big for OCR."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _extract_texts_scores(result):
    """Parse OCRResult and return (texts, scores)."""
    texts, scores = [], []
    if result and result[0]:
        res = result[0]
        try:
            texts = res["rec_texts"]
            scores = res["rec_scores"]
        except (KeyError, TypeError):
            texts = getattr(res, "rec_texts", [])
            scores = getattr(res, "rec_scores", [])
    texts = [t for t in (texts or []) if t.strip()]
    scores = list(scores or [])
    return texts, scores


def _looks_valid(texts: list, scores: list) -> bool:
    """Check if OCR output contains recognizable document patterns."""
    if not texts:
        return False
    raw = " ".join(texts)
    for pat in _VALID_PATTERNS:
        if pat.search(raw):
            return True
    if scores and float(np.mean(scores)) > 0.85:
        return True
    return False


def paddle_ocr_and_annotate(img_path: str, ocr=ocr):
    start_time = time.time()

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Step 1: Downscale to 1000px max (much faster OCR)
    img = _resize(img)

    # Step 2: First OCR attempt
    texts, scores = _extract_texts_scores(ocr.ocr(img))

    # Step 3: If text looks garbled, try 180° rotation
    if not _looks_valid(texts, scores):
        img_180 = cv2.rotate(img, cv2.ROTATE_180)
        texts_180, scores_180 = _extract_texts_scores(ocr.ocr(img_180))
        if _looks_valid(texts_180, scores_180):
            texts = texts_180
        elif scores_180 and (not scores or float(np.mean(scores_180)) > float(np.mean(scores))):
            texts = texts_180

    raw_text = " ".join(texts)

    return {
        "texts": texts,
        "raw_text": raw_text,
        "annotated_path": None,
        "execution_time": round(time.time() - start_time, 3)
    }
