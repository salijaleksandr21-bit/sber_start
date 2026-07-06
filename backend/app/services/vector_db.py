import faiss
import json
import os
from sentence_transformers import SentenceTransformer

class VectorDBService:
    def __init__(self):
        print("⏳ Инициализация векторной БД...")
        
        # Вычисляем пути
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_dir)) 
        index_path = os.path.join(backend_dir, "data", "faiss_index.bin")
        cases_path = os.path.join(backend_dir, "data", "mock_cases.json")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Файл индекса не найден по пути: {index_path}")

        # Загрузка
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self.index = faiss.read_index(index_path)
        with open(cases_path, "r", encoding="utf-8") as f:
            self.cases = json.load(f)
            
        print("✅ Векторная БД готова к работе")

    # ВАЖНО: Вот эта функция потерялась в прошлом шаге!
    def search_similar(self, query: str, k: int = 2):
        # Префикс 'query: ' рекомендуется для моделей e5
        query_vector = self.model.encode([f"query: {query}"]) 
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            results.append(self.cases[idx])
        return results