import pandas as pd
import faiss
import json
import os
from sentence_transformers import SentenceTransformer

# 1. Сбор синтетического датасета (имитация PostgreSQL данных)
mock_data = [
    {
        "id": 1,
        "niche": "кофейня",
        "description": "Кофейня формата to-go возле университета. Целевая аудитория: студенты.",
        "success_rate_percent": 65,
        "common_risk": "Кассовый разрыв летом (отсутствие студентов).",
        "survival_12_months": True
    },
    {
        "id": 2,
        "niche": "кофейня",
        "description": "Премиум кофейня с авторской выпечкой в спальном районе.",
        "success_rate_percent": 30,
        "common_risk": "Несовпадение ценового ожидания аудитории спального района.",
        "survival_12_months": False
    },
    {
        "id": 3,
        "niche": "IT-услуги",
        "description": "Агентство по разработке лендингов и сайтов на Tilda для малого бизнеса.",
        "success_rate_percent": 80,
        "common_risk": "Высокая конкуренция, демпинг цен фрилансерами.",
        "survival_12_months": True
    },
    {
        "id": 4,
        "niche": "одежда",
        "description": "Бренд уличной одежды с продажей через Instagram и маркетплейсы.",
        "success_rate_percent": 45,
        "common_risk": "Сложности с логистикой и браком на производстве.",
        "survival_12_months": False
    }
]

# Создаем папку data, если ее нет
os.makedirs("../data", exist_ok=True)

# Сохраняем "БД" в JSON (в будущем это будет PostgreSQL)
with open("../data/mock_cases.json", "w", encoding="utf-8") as f:
    json.dump(mock_data, f, ensure_ascii=False, indent=4)

print("✅ Датасет успешно создан.")

# 2. Настройка модели эмбеддингов
print("⏳ Загрузка модели эмбеддингов (это может занять пару минут)...")
# Берем легкую модель для тестов. Можно заменить на intfloat/multilingual-e5-large
model = SentenceTransformer('intfloat/multilingual-e5-large') 

# Формируем тексты для векторизации (именно по ним будем искать сходство)
texts_to_embed = [case["description"] for case in mock_data]

print("⏳ Векторизация описаний бизнесов...")
embeddings = model.encode(texts_to_embed)

# 3. Настройка и загрузка данных в FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension) # Используем L2 дистанцию для поиска
index.add(embeddings)

faiss.write_index(index, "../data/faiss_index.bin")
print(f"✅ Векторный индекс FAISS создан и сохранен. Размерность вектора: {dimension}, количество кейсов: {index.ntotal}")

# 4. Тестирование векторного поиска
query = "Хочу открыть точку с кофе рядом с ВШЭ"
query_vector = model.encode([query])

k = 2 # Ищем 2 ближайших соседа
distances, indices = index.search(query_vector, k)

print(f"\n🔍 Тест поиска для запроса: '{query}'")
for i in range(k):
    case_idx = indices[0][i]
    print(f"[{i+1}] Найдено совпадение (ID: {mock_data[case_idx]['id']}): {mock_data[case_idx]['description']} (Дистанция: {distances[0][i]:.4f})")