"""
TrustLens — OCR Module (Model 3, IMAGE data-space)
---------------------------------------------------
Real users are frequently scammed via *screenshots* — a forwarded WhatsApp
image of a "bank alert", a photo of a fake login page, etc. Pure text tools
cannot see these. This module lifts text out of an uploaded image so it can
re-enter the *same* text-analysis pipeline (classifier + emotion + indicators).

This is the component that makes TrustLens genuinely multimodal and satisfies
the CM3020 4.1 requirement of pre-trained models across *different data spaces*
(text vs image), rather than several text models.

Engine: Tesseract OCR (via pytesseract). Tesseract is a mature, pre-trained
optical-character-recognition engine. It needs the system binary `tesseract`
installed (e.g. `brew install tesseract` on macOS, `apt-get install tesseract-ocr`
on Debian/Ubuntu) plus the Python packages `pytesseract` and `Pillow`.

Graceful degradation: if the engine or its Python bindings are missing, the
module returns available=False with a helpful message instead of crashing.
"""

from io import BytesIO


def _engine_ready() -> bool:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        # Probe the underlying binary; raises if tesseract is not on PATH.
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Run OCR on raw image bytes and return the recognised text.

    Returns
    -------
    dict with keys:
        available   : bool  — whether the OCR engine ran
        text        : str   — extracted text (may be empty)
        char_count  : int
        note        : str   — human-readable status / guidance
    """
    if not _engine_ready():
        return {
            "available": False,
            "text": "",
            "char_count": 0,
            "note": ("OCR engine not installed. Install the Tesseract binary and "
                     "`pip install pytesseract Pillow` to enable screenshot analysis."),
        }

    try:
        import pytesseract
        from PIL import Image, ImageOps

        img = Image.open(BytesIO(image_bytes))
        # Light preprocessing improves OCR on phone screenshots: greyscale + autocontrast.
        img = ImageOps.grayscale(img)
        img = ImageOps.autocontrast(img)

        text = pytesseract.image_to_string(img).strip()
        return {
            "available": True,
            "text": text,
            "char_count": len(text),
            "note": "OCR succeeded." if text else "OCR ran but found no readable text.",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "text": "",
            "char_count": 0,
            "note": f"OCR failed: {type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    # Smoke test: build a small image with text so this runs without a real screenshot.
    try:
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (600, 120), "white")
        ImageDraw.Draw(im).text((10, 40),
                                "Your account is locked. Verify your password now.",
                                fill="black")
        buf = BytesIO(); im.save(buf, format="PNG")
        print(extract_text_from_image(buf.getvalue()))
    except Exception as e:  # noqa: BLE001
        print("Smoke test could not run:", e)
