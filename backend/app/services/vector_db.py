import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDBService:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        # Определяем корневую папку проекта (backend)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = os.path.join(base_dir, data_dir)
        
        print("⏳ Загрузка модели эмбеддингов...")
        self.model = SentenceTransformer('intfloat/multilingual-e5-large')
        self._load_index_and_cases()

    def _load_index_and_cases(self):
        index_path = os.path.join(self.data_dir, "faiss_index.bin")
        cases_path = os.path.join(self.data_dir, "mock_cases.json")
        if not os.path.exists(index_path) or not os.path.exists(cases_path):
            raise FileNotFoundError(
                f"Файлы индекса или кейсов не найдены в {self.data_dir}. "
                "Сначала запусти скрипт init_db_and_faiss.py"
            )
        self.index = faiss.read_index(index_path)
        with open(cases_path, "r", encoding="utf-8") as f:
            self.cases = json.load(f)
        print(f"✅ Загружено {len(self.cases)} кейсов, размерность индекса: {self.index.d}")

    def search_similar(self, query_text: str, k: int = 30) -> list:
        # Защита от падения FAISS: запрашиваем не больше, чем есть в базе
        actual_k = min(k, self.index.ntotal)
        if actual_k == 0:
            return []
            
        # Для модели e5-large рекомендуется префикс "query: "
        vector = self.model.encode([f"query: {query_text}"])
        distances, indices = self.index.search(vector.astype(np.float32), actual_k)
        
        # Убираем -1 (если индекс вернул меньше k)
        valid_indices = [idx for idx in indices[0] if idx != -1]
        return [self.cases[idx] for idx in valid_indices]