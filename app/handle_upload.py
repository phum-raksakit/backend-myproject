from fastapi import UploadFile
from pathlib import Path
import shutil
from uuid import uuid4
from model import Model
from model2 import Model2
import fitz

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def handle_upload(file: UploadFile):
    ext = Path(file.filename).suffix
    safe_name = f"{uuid4().hex}{ext}"
    dest = UPLOAD_DIR / safe_name

    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


    doc = fitz.open(dest)
    # model = Model()
    # fields = model.processDocument(doc)

    model = Model2()
    poppler = r"C:\poppler-25.07.0\Library\bin"
    fields = model.processDocumentOCR(str(dest), poppler_path=poppler)
    field_report = model.check_fields(fields)

    for fname, info in field_report.items():
        print(f"== {fname} ==")
        print("status:", info["status"])
        if info["errors"]:
            for tok, sugg in info["errors"]:
                print("  error:", tok, "->", sugg)
        if info["unknowns"]:
            print("  unknowns:", info["unknowns"])


    return {
        "status": "ok",
        "filename": file.filename,
        "saved_as": safe_name,
        "fields": fields,
    }
