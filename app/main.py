from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from handle_upload import handle_upload

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    result = await handle_upload(file)
    return JSONResponse(result)
