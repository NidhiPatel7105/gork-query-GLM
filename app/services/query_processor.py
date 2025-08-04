import json
import os
from xai import Client
from typing import Dict, Any

class FineTunedQueryProcessor:
    def __init__(self):
        self.client = Client(api_key=os.getenv("GROK_API_KEY"))
        self.intent_prompt = """
        Analyze the following query and extract:
        1. The main intent or question being asked
        2. Key entities or terms that are important for retrieval
        3. The type of information being requested
        
        Query: {query}
        
        Return a JSON object with the following structure:
        {{
            "intent": "Brief description of the main intent",
            "entities": ["entity1", "entity2", ...],
            "information_type": "Type of information requested"
        }}
        """
    
    async def extract_query_intent(self, query: str) -> Dict[str, Any]:
        domain = self._detect_domain(query)
        
        prompt = self.intent_prompt.format(query=query)
        
        response = self.client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": f"You are an expert at analyzing queries for {domain} document retrieval systems."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            result = response.choices[0].message.content
            return json.loads(result)
        except json.JSONDecodeError:
            return {
                "intent": query,
                "entities": [],
                "information_type": "general"
            }
    
    def _detect_domain(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["policy", "coverage", "premium", "insurance", "claim"]):
            return "insurance"
        elif any(word in query_lower for word in ["contract", "agreement", "clause", "legal", "law"]):
            return "legal"
        elif any(word in query_lower for word in ["employee", "hr", "human resources", "leave", "termination"]):
            return "hr"
        elif any(word in query_lower for word in ["compliance", "regulation", "audit", "standard"]):
            return "compliance"
        else:
            return "general"