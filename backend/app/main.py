from fastapi import FastAPI
import uvicorn
from app.schemas.pydantic_models import IdeaRequest, AnalysisResponse
from app.services.vector_db import VectorDBService
from app.services.llm_service import LLMService
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Сбер.Старт API", version="1.0.0")

# Инициализируем сервисы при старте
vector_db = VectorDBService()
llm_service = LLMService()

# Создаем абсолютный путь к папке static
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
static_dir = os.path.join(backend_dir, "static")

# Создаем папку, если ее нет
os.makedirs(static_dir, exist_ok=True)

# Подключаем раздачу статических файлов
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Инициализируем сервисы (остается как было)
vector_db = VectorDBService()
llm_service = LLMService()

# Новый эндпоинт для главной страницы
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return "<h1>Файл index.html не найден в папке static!</h1>"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_idea(request: IdeaRequest):
    # 1. Поиск аналогов (Vector Search)
    similar_cases = vector_db.search_similar(request.idea_text)
    
    # 2. Агрегация данных (Backend)
    avg_survival = sum(case['success_rate_percent'] for case in similar_cases) / len(similar_cases)
    
    # 3. Генерация ответа (LLM 2)
    llm_analysis = llm_service.generate_advice(request.idea_text, similar_cases)
    
    # Формируем итоговый ответ
    return AnalysisResponse(
        niche=similar_cases[0]["niche"], # Берем нишу из самого близкого кейса
        similar_cases_count=len(similar_cases),
        survival_probability=avg_survival,
        key_risks=llm_analysis.get("key_risks", []),
        recommendations=llm_analysis.get("recommendations", []),
        summary=llm_analysis.get("summary", "")
    )
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)