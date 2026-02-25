import re

DOCUMENT_PATTERNS = {
    "PAN": r"[A-Z]{5}[0-9]{4}[A-Z]",
    "Aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "Driving_License": r"[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}",
    "Passport": r"\b[A-Z]\d{7}\b",
    "UDYAM": r"UDYAM-[A-Z]{2}-\d{2}-\d{7}",
}

def extract_document_ids(text: str):
    found = {}
    for doc, pattern in DOCUMENT_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[doc] = list(set(matches))
    return found
