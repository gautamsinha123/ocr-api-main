import os
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# use_doc_orientation_classify=False → we handle orientation manually (more reliable + faster)
# use_doc_unwarping=False            → skips UVDoc (~15s saved)
# use_textline_orientation=False     → skips per-line orientation (~5s saved)
ocr = PaddleOCR(
    lang="en",
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    use_textline_orientation=False,
)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Auto-detect and correct skew angle using Hough line detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (card edges)
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    skew = float(np.median(angles))
    # Only correct if skew is significant (>1°) and not extreme (>30°)
    if abs(skew) < 1.0 or abs(skew) > 30.0:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), skew, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _scale_for_orient(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Downscale to max_dim on longest side for fast orientation detection."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _enhance(img: np.ndarray) -> np.ndarray:
    """Smart resize + sharpen + denoise for best OCR accuracy."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    # Phone photos (>1500px) are already high-res — no upscale needed
    # Small images (<800px) benefit from 2x upscale
    if max_dim < 800:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    elif max_dim < 1500:
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    # else: large image — keep original size
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    img = cv2.bilateralFilter(img, d=3, sigmaColor=30, sigmaSpace=30)
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


def paddle_ocr_and_annotate(img_path: str, ocr=ocr):
    start_time = time.time()

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Step 1: Deskew at original resolution (correct tilt)
    img = _deskew(img)

    # Step 2: Fast orientation check on a small (≤800px) version — 2 quick OCR calls
    small = _scale_for_orient(img)
    _, scores_0 = _extract_texts_scores(ocr.ocr(small))
    _, scores_180 = _extract_texts_scores(ocr.ocr(cv2.rotate(small, cv2.ROTATE_180)))

    avg_0   = float(np.mean(scores_0))   if scores_0   else 0.0
    avg_180 = float(np.mean(scores_180)) if scores_180 else 0.0

    # Step 3: Rotate full image to correct orientation if needed
    if avg_180 > avg_0:
        img = cv2.rotate(img, cv2.ROTATE_180)

    # Step 4: Enhance at appropriate resolution (smart scale + sharpen + denoise)
    img_final = _enhance(img)

    # Step 5: Single full-quality OCR call on the correctly oriented image
    texts, _ = _extract_texts_scores(ocr.ocr(img_final))

    raw_text = " ".join(texts)

    return {
        "texts": texts,
        "raw_text": raw_text,
        "annotated_path": None,
        "execution_time": round(time.time() - start_time, 3)
    }
