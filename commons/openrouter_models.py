#!/usr/bin/env python3
"""Получение и анализ списка моделей OpenRouter."""
import requests
from datetime import datetime

API_URL = "https://openrouter.ai/api/v1/models"

def fetch_models():
    """Загрузить список моделей."""
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    return r.json()["data"]

def format_price(price_str: str) -> str:
    """Конвертировать цену за токен в цену за 1M токенов."""
    try:
        price_per_1m = float(price_str) * 1_000_000
        return f"${price_per_1m:.2f}"
    except (ValueError, TypeError):
        return "N/A"

def main():
    models = fetch_models()
    print(f"✅ Всего моделей: {len(models)}")
    print(f"📅 Обновлено: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Фильтр: только модели, которые нас интересуют
    keywords = ["claude", "gemini", "deepseek", "qwen", "gpt", "grok", "glm"]
    filtered = [
        m for m in models
        if any(kw in m["id"].lower() for kw in keywords)
    ]

    # Сортировка по цене input
    filtered.sort(key=lambda m: float(m.get("pricing", {}).get("prompt", "999") or "999"))

    # Печать таблицы
    print(f"{'MODEL ID':<55} {'INPUT/1M':>10} {'OUTPUT/1M':>10} {'CONTEXT':>10}")
    print("-" * 90)
    for m in filtered:
        pricing = m.get("pricing", {})
        print(
            f"{m['id']:<55} "
            f"{format_price(pricing.get('prompt')):>10} "
            f"{format_price(pricing.get('completion')):>10} "
            f"{m.get('context_length', 'N/A'):>10}"
        )

if __name__ == "__main__":
    main()