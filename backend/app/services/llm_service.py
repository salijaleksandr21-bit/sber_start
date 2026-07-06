import requests
import json
from typing import List, Dict

class LLMService:
    def __init__(self):
        # Локальный URL Ollama
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model_name = "qwen2.5:7b"

    def generate_advice(self, user_idea: str, similar_cases: List[Dict]):
        # Агрегируем данные для промпта
        avg_survival = sum(case['success_rate_percent'] for case in similar_cases) / len(similar_cases)
        risks = "\n".join([f"- {case['common_risk']}" for case in similar_cases])
        
        prompt = f"""
        Ты - строгий, но полезный AI-аналитик банка "Сбер". 
        Пользователь хочет открыть бизнес: "{user_idea}"
        
        Мы нашли похожие исторические кейсы. 
        Средняя выживаемость таких проектов: {avg_survival}%
        Частые риски из практики:
        {risks}
        
        На основе этих данных напиши краткий JSON-ответ (СТРОГО валидный JSON).
        Ключи JSON:
        "summary": Краткий вывод о жизнеспособности.
        "key_risks": Массив из 3 главных рисков.
        "recommendations": Массив из 3 действий перед стартом.
        """

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json", # Заставляем Ollama выдать JSON
            "stream": False
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result_text = response.json()["response"]
            return json.loads(result_text)
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return {
                "summary": "Ошибка анализа ИИ.",
                "key_risks": ["Не удалось получить данные"],
                "recommendations": ["Попробуйте позже"]
            }