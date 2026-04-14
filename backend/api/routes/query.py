from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.models.query import QueryRequest
from backend.services.query_service import answer_query

router = APIRouter()


@router.post("/")
async def query(request: QueryRequest):
    async def event_stream():
        async for token in answer_query(request):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
