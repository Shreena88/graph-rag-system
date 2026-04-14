import os, tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.models.document import DocumentResult
from backend.services import ingestion_service
from backend.config import settings

router = APIRouter()


@router.post("/upload", response_model=DocumentResult)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if file.size and file.size > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, "File too large")

    allowed = {".pdf", ".txt", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Register doc immediately so status endpoint works, then ingest in background
    doc_id = await ingestion_service.register_document(file.filename)
    background_tasks.add_task(ingestion_service.ingest_document, tmp_path, file.filename, doc_id)
    result = await ingestion_service.get_status(doc_id)
    if result is None:
        # Redis unavailable - return status directly
        from backend.models.document import DocumentResult
        result = DocumentResult(doc_id=doc_id, filename=file.filename, status="processing")
    return result


@router.get("/{doc_id}/status", response_model=DocumentResult)
async def get_status(doc_id: str):
    result = await ingestion_service.get_status(doc_id)
    if not result:
        raise HTTPException(404, "Document not found")
    return result
