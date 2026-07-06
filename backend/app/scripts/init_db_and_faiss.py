import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Определяем корень проекта (папка backend)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
cases_path = os.path.join(data_dir, "mock_cases.json")
index_path = os.path.join(data_dir, "faiss_index.bin")

# 1. Сбор синтетического датасета (имитация PostgreSQL)
mock_data = [
    {
        "id": 1,
        "niche": "кофейня",
        "description": "Кофейня формата to-go возле университета. Целевая аудитория: студенты.",
        "success_rate_percent": 65,
        "common_risk": "Кассовый разрыв летом (отсутствие студентов)",
        "top_risks": ["Кассовый разрыв летом", "Высокая аренда", "Низкая маржинальность"],
        "success_factors": ["Удачная локация", "Акцент на качестве", "Программа лояльности"],
        "survival_12_months": True
    },
    {
        "id": 2,
        "niche": "кофейня",
        "description": "Премиум кофейня с авторской выпечкой в спальном районе.",
        "success_rate_percent": 30,
        "common_risk": "Несовпадение ценового ожидания аудитории спального района",
        "top_risks": ["Ценовая политика", "Отсутствие постоянных клиентов", "Сезонность"],
        "success_factors": ["Уникальный ассортимент", "Качество", "Атмосфера"],
        "survival_12_months": False
    },
    {
        "id": 3,
        "niche": "IT-услуги",
        "description": "Агентство по разработке лендингов и сайтов на Tilda для малого бизнеса.",
        "success_rate_percent": 80,
        "common_risk": "Высокая конкуренция, демпинг цен фрилансерами",
        "top_risks": ["Демпинг", "Текучесть кадров", "Нестабильный поток заказов"],
        "success_factors": ["Хорошие портфолио", "Быстрая обратная связь", "Гибкие цены"],
        "survival_12_months": True
    },
    {
        "id": 4,
        "niche": "одежда",
        "description": "Бренд уличной одежды с продажей через Instagram и маркетплейсы.",
        "success_rate_percent": 45,
        "common_risk": "Сложности с логистикой и браком на производстве",
        "top_risks": ["Логистика", "Брак", "Большие возвраты"],
        "success_factors": ["Уникальный дизайн", "Активный SMM", "Работа с инфлюенсерами"],
        "survival_12_months": False
    },
    # Добавь больше кейсов для разнообразия (здесь я для примера даю только 4, но ты можешь расширить до 100+)
]

# Сохраняем как JSON
with open(cases_path, "w", encoding="utf-8") as f:
    json.dump(mock_data, f, ensure_ascii=False, indent=4)
print(f"✅ Датасет сохранён в {cases_path} (записей: {len(mock_data)})")

# 2. Настройка модели эмбеддингов
print("⏳ Загрузка модели эмбеддингов...")
model = SentenceTransformer('intfloat/multilingual-e5-large')
texts_to_embed = [case["description"] for case in mock_data]
print("⏳ Векторизация...")
embeddings = model.encode(texts_to_embed)

# 3. Создание FAISS индекса
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype(np.float32))
faiss.write_index(index, index_path)
print(f"✅ Индекс сохранён в {index_path}, размерность: {dimension}, кейсов: {index.ntotal}")

# 4. Тест поиска
query = "Хочу открыть точку с кофе рядом с ВШЭ"
query_vector = model.encode([f"query: {query}"])
distances, indices = index.search(query_vector.astype(np.float32), 2)
print(f"\n🔍 Тест поиска для запроса: '{query}'")
for i, idx in enumerate(indices[0]):
    if idx != -1:
        print(f"[{i+1}] {mock_data[idx]['description']} (расстояние: {distances[0][i]:.4f})")