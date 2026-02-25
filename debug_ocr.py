import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang="en")

def try_ocr(img, label):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(np.array(rgb))
    print(f"\n{'='*40}")
    print(f"CANDIDATE: {label}")
    print(f"{'='*40}")
    if result and result[0]:
        res = result[0]
        try:
            texts  = res["rec_texts"]
            scores = res["rec_scores"]
        except (KeyError, TypeError):
            texts  = getattr(res, "rec_texts", [])
            scores = getattr(res, "rec_scores", [])
        for txt, scr in zip(texts, scores):
            if txt.strip():
                print(f"  [{scr:.2f}] {txt}")
    else:
        print("  NO TEXT DETECTED")

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

# ---- UPDATE THIS PATH TO YOUR p2.jpeg ----
img_path = r"C:\Users\VSK\Downloads\p2.jpeg"
# ------------------------------------------

img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Could not read image at: {img_path}")
    print("Update the img_path variable to the correct location of p2.jpeg")
    exit()

print(f"Image loaded: {img.shape[1]}x{img.shape[0]}")
upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

print("\n--- Testing base orientations (upscaled) ---")
try_ocr(upscaled,                                        "Original x2")
try_ocr(cv2.flip(upscaled, 1),                           "HFlip x2")
try_ocr(cv2.rotate(upscaled, cv2.ROTATE_180),            "Rotate180 x2")
try_ocr(cv2.flip(cv2.rotate(upscaled,cv2.ROTATE_180),1), "Rotate180+HFlip x2")

print("\n--- Testing fine angles on HFlip x2 ---")
base = cv2.flip(upscaled, 1)
for angle in [-20, -15, -12, -10, -8, -5, -3, 0, 3, 5, 8, 10, 12, 15, 20]:
    try_ocr(rotate_image(base, angle), f"HFlip + {angle}deg")

print("\nDONE")
