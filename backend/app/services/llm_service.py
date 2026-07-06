import requests
import json
from typing import List, Dict

class LLMService:
    def __init__(self, model_name: str = "qwen2.5:7b", ollama_url: str = "http://localhost:11434/api/generate"):
        self.ollama_url = ollama_url
        self.model_name = model_name

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.3}
        }
        response = requests.post(self.ollama_url, json=payload)
        response.raise_for_status()
        return response.json()["response"]

    def generate_advice(self, user_idea: str, aggregated: Dict) -> Dict:
        """
        Принимает агрегированную статистику и генерирует JSON-ответ с рекомендациями.
        """
        prompt = f"""
        Ты - строгий, но полезный AI-аналитик банка "Сбер". 
        Пользователь хочет открыть бизнес: "{user_idea}"
        
        Мы нашли {aggregated['total_cases']} похожих бизнесов.
        Средняя выживаемость: {aggregated['survival_rate']:.1f}%
        Основные риски: {', '.join(aggregated['top_risks'])}
        Факторы успеха: {', '.join(aggregated['success_factors'])}
        
        На основе этих данных напиши краткий JSON-ответ (СТРОГО валидный JSON).
        Ключи JSON:
        "summary": Краткий вывод о жизнеспособности (1-2 предложения).
        "key_risks": Массив из 3 главных рисков (строки).
        "recommendations": Массив из 3 действий перед стартом (строки).
        """
        try:
            result_text = self._call_ollama(prompt)
            return json.loads(result_text)
        except Exception as e:
            print(f"Ошибка LLM: {e}")
            return {
                "summary": "Не удалось получить анализ от ИИ. Используем общую статистику.",
                "key_risks": aggregated.get("top_risks", ["Недостаточно данных"]),
                "recommendations": ["Проведите более детальный анализ рынка"]
            }

    def extract_profile(self, idea_text: str) -> dict:
        prompt = f"""
        Проанализируй бизнес-идею и верни СТРОГО валидный JSON (без лишних слов).
        Поля: "niche" (ниша), "target_audience" (ЦА), "scale" (масштаб), "format" (формат: онлайн/офлайн).
        Идея: {idea_text}
        """
        try:
            result = self._call_ollama(prompt)
            return json.loads(result)
        except Exception as e:
            print(f"Ошибка профилирования: {e}")
            # Фолбэк, если ИИ ошибся в формате
            return {"niche": "неизвестно", "target_audience": "масс-маркет", "scale": "микробизнес", "format": "смешанный"}