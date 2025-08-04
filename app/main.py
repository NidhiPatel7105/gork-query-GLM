from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from redis import asyncio as aioredis
from pydantic import BaseModel
from typing import List
import os
from app.config import settings
from app.services.document_processor import DocumentProcessor
from app.services.embedding_manager import EmbeddingManager
from app.services.query_processor import FineTunedQueryProcessor
from app.services.answer_generator import LogicEvaluator, AnswerGenerator

app = FastAPI()
security = HTTPBearer()

class SubmissionRequest(BaseModel):
    documents: str
    questions: List[str]

class SubmissionResponse(BaseModel):
    answers: List[str]

# Initialize services
document_processor = DocumentProcessor()
embedding_manager = EmbeddingManager()
query_processor = FineTunedQueryProcessor()
logic_evaluator = LogicEvaluator()
answer_generator = AnswerGenerator()

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url(settings.redis_url)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

@app.post("/hackrx/run", response_model=SubmissionResponse)
@cache(expire=1800)  # Cache for 30 minutes
async def run_submission(
    request: SubmissionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate token
    if credentials.credentials != settings.api_token:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    try:
        # Process document
        doc_id, chunks = await document_processor.process_document(request.documents)
        
        # Generate embeddings and upsert
        await embedding_manager.upsert_embeddings(doc_id, chunks)
        
        answers = []
        for question in request.questions:
            # Extract query intent
            intent = await query_processor.extract_query_intent(question)
            
            # Search similar clauses
            similar_clauses = await embedding_manager.search_similar(question, doc_id)
            
            # Evaluate logic
            evaluation = await logic_evaluator.evaluate_logic(question, similar_clauses)
            
            # Generate answer
            answer = await answer_generator.generate_answer(question, evaluation, similar_clauses)
            answers.append(answer)
        
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")