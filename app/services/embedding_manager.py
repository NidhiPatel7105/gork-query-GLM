import pinecone
from openai import AsyncOpenAI
from typing import List, Dict, Any
from app.config import settings

class EmbeddingManager:
    def __init__(self):
        pinecone.init(
            api_key=settings.pinecone_api_key,
            environment=settings.pinecone_environment
        )
        self.index_name = settings.pinecone_index_name
        self.index = pinecone.Index(self.index_name)
        self.embedding_model = "text-embedding-ada-002"
        self.client = AsyncOpenAI(
            base_url="https://api.x.ai/v1",
            api_key=settings.grok_api_key
        )
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [data.embedding for data in response.data]
    
    async def upsert_embeddings(self, doc_id: str, chunks: List[Dict[str, Any]]):
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await self.generate_embeddings(texts)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{doc_id}-{i}"
            metadata = chunk["metadata"].copy()
            metadata["content"] = chunk["content"]
            metadata["doc_id"] = doc_id
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        self.index.upsert(vectors=vectors)
    
    async def search_similar(self, query: str, doc_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = (await self.generate_embeddings([query]))[0]
        
        filter_dict = {}
        if doc_id:
            filter_dict["doc_id"] = {"$eq": doc_id}
        
        results = self.index.query(
            vector=query_embedding,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True
        )
        
        matches = []
        for match in results["matches"]:
            matches.append({
                "id": match["id"],
                "score": match["score"],
                "content": match["metadata"]["content"],
                "metadata": match["metadata"]
            })
        
        return matches