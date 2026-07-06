from pydantic import BaseModel, Field
from typing import List

class IdeaRequest(BaseModel):
    idea_text: str = Field(..., example="Хочу открыть кофейню возле ВШЭ")

class AnalysisResponse(BaseModel):
    niche: str
    survival_probability: float
    similar_cases_count: int
    key_risks: List[str]
    recommendations: List[str]
    summary: str