import os
import tempfile
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from backend.pdf_metadata_extractor import PDFMetadataExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFMetadataAPI")

app = FastAPI(title="PDF Metadata Analysis API", version="1.0")

extractor = PDFMetadataExtractor()


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF and returns everything extractable about the FILE itself
    (metadata, structure/revision signals, text layer, signatures, embedded files,
    XMP) — no OCR, no fraud verdict. See backend/pdf_metadata_extractor.py.
    """
    is_pdf = (file.content_type == "application/pdf") or (file.filename or "").lower().endswith(".pdf")
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    temp_dir = tempfile.gettempdir()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=temp_dir)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())

        result = extractor.analyze(temp_file_path)
        return JSONResponse(content={**result, "file_name": file.filename})

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"System Error: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
