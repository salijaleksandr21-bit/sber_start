from pydantic import BaseModel
from typing import List, Optional

class IdeaRequest(BaseModel):
    idea_text: str

class AnalysisResponse(BaseModel):
    niche: str
    similar_cases_count: int
    survival_probability: float
    key_risks: List[str]
    recommendations: List[str]
    summary: str
    # Можно добавить дополнительные поля, если нужно
    # profile: Optional[dict] = None