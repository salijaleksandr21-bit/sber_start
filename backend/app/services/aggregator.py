from collections import Counter
from typing import List, Dict

class AggregatorService:
    @staticmethod
    def aggregate(cases: List[Dict]) -> Dict:
        if not cases:
            return {
                "total_cases": 0,
                "survival_rate": 0.0,
                "top_risks": [],
                "success_factors": [],
                "most_common_niche": "неизвестно"
            }
        total = len(cases)
        survival_sum = sum(c.get("success_rate_percent", 0) for c in cases)
        avg_survival = survival_sum / total

        # Собираем все риски (из поля top_risks, если есть, иначе из common_risk)
        all_risks = []
        for c in cases:
            if "top_risks" in c and isinstance(c["top_risks"], list):
                all_risks.extend(c["top_risks"])
            elif "common_risk" in c:
                all_risks.append(c["common_risk"])
        risk_counts = Counter(all_risks)
        top_risks = [item for item, _ in risk_counts.most_common(3)]

        # Аналогично для факторов успеха
        all_factors = []
        for c in cases:
            if "success_factors" in c and isinstance(c["success_factors"], list):
                all_factors.extend(c["success_factors"])
        factor_counts = Counter(all_factors)
        top_factors = [item for item, _ in factor_counts.most_common(3)]

        # Самая частая ниша
        niches = [c.get("niche", "неизвестно") for c in cases]
        most_common_niche = Counter(niches).most_common(1)[0][0]

        return {
            "total_cases": total,
            "survival_rate": avg_survival,
            "top_risks": top_risks,
            "success_factors": top_factors,
            "most_common_niche": most_common_niche
        }