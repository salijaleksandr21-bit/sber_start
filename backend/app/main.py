from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os
import sys

# Добавляем путь для импорта (на случай, если запускаем не из корня)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import IdeaRequest, AnalysisResponse
from app.services import VectorDBService, LLMService, AggregatorService

app = FastAPI(title="Сбер.Старт MVP", version="1.0.0")

# Инициализация сервисов
vector_db = VectorDBService(data_dir="data")
llm_service = LLMService()
aggregator = AggregatorService()

# Подключаем статику
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>Файл index.html не найден в static/</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_idea(request: IdeaRequest):
    idea_text = request.idea_text

    # 1. Формируем цифровой профиль через LLM
    profile_dict = llm_service.extract_profile(idea_text)
    
    # Склеиваем значения профиля в осмысленную строку для FAISS
    profile_str = " ".join([str(v) for v in profile_dict.values()])
    print(f"DEBUG: Ищем по профилю: {profile_str}")

    # 2. Поиск похожих кейсов по профилю (до 30 штук, FAISS защищен внутри сервиса)
    similar_cases = vector_db.search_similar(profile_str, k=30)
    
    if not similar_cases:
        return AnalysisResponse(
            niche="неизвестно",
            similar_cases_count=0,
            survival_probability=0.0,
            key_risks=["Недостаточно данных"],
            recommendations=["Попробуйте описать идею подробнее"],
            summary="Не найдено похожих бизнесов в базе."
        )

    # 3. Агрегация
    aggregated = aggregator.aggregate(similar_cases)

    # 4. Генерация рекомендаций через LLM
    llm_result = llm_service.generate_advice(idea_text, aggregated)

    return AnalysisResponse(
        niche=aggregated["most_common_niche"],
        similar_cases_count=aggregated["total_cases"],
        survival_probability=round(aggregated["survival_rate"], 1),
        key_risks=llm_result.get("key_risks", aggregated["top_risks"]),
        recommendations=llm_result.get("recommendations", []),
        summary=llm_result.get("summary", "")
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)