import json
import os
from xai import Client
from typing import List, Dict, Any

class LogicEvaluator:
    def __init__(self):
        self.client = Client(api_key=os.getenv("GROK_API_KEY"))
        self.evaluation_prompt = """
        Based on the following query and relevant document clauses, evaluate the logic and determine the answer.
        
        Query: {query}
        
        Relevant Clauses:
        {clauses}
        
        Provide a detailed evaluation that includes:
        1. A direct answer to the query
        2. Reasoning for your answer based on the clauses
        3. Any conditions or limitations mentioned in the clauses
        4. A confidence score (0-1) indicating how certain you are of the answer
        
        Return a JSON object with the following structure:
        {{
            "answer": "Direct answer to the query",
            "reasoning": "Detailed reasoning for the answer",
            "conditions": ["Condition 1", "Condition 2", ...],
            "confidence": 0.95
        }}
        """
    
    async def evaluate_logic(self, query: str, clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        clauses_text = "\n\n".join([f"Clause {i+1}: {clause['content']}" for i, clause in enumerate(clauses)])
        
        prompt = self.evaluation_prompt.format(
            query=query,
            clauses=clauses_text
        )
        
        response = self.client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing insurance policies and legal documents to provide accurate answers."},
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
                "answer": "Unable to determine answer from provided clauses.",
                "reasoning": "Error processing the clauses.",
                "conditions": [],
                "confidence": 0.0
            }

class AnswerGenerator:
    def __init__(self):
        self.client = Client(api_key=os.getenv("GROK_API_KEY"))
        self.answer_prompt = """
        Generate a comprehensive answer to the following query based on the evaluation result.
        
        Query: {query}
        
        Evaluation Result:
        {evaluation}
        
        Generate a response that:
        1. Directly answers the query
        2. Includes any relevant conditions or limitations
        3. Is clear, concise, and easy to understand
        4. Cites the relevant clauses that support the answer
        """
    
    async def generate_answer(self, query: str, evaluation: Dict[str, Any], clauses: List[Dict[str, Any]]) -> str:
        evaluation_text = f"""
        Answer: {evaluation['answer']}
        Reasoning: {evaluation['reasoning']}
        Conditions: {', '.join(evaluation['conditions'])}
        Confidence: {evaluation['confidence']}
        """
        
        prompt = self.answer_prompt.format(
            query=query,
            evaluation=evaluation_text
        )
        
        response = self.client.chat.completions.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are an expert at generating clear and comprehensive answers based on document analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        return response.choices[0].message.content