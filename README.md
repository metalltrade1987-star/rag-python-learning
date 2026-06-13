# 🤖 Локальная RAG-система для поиска по документации

> Учебный проект, демонстрирующий полный цикл создания RAG-системы: от индексации текста до оценки качества. Оптимизирован для работы на устройствах с 8 ГБ ОЗУ без облачных API.

## 📊 Архитектура

```mermaid
graph LR
  A[knowledge.txt] --> B(Chunking & Embedding)
  B --> C[ChromaDB]
  D[User Query] --> E(Embed Query)
  E --> C
  C --> F[Top-K Retrieval]
  F --> G[Reranking / Scoring]
  G --> H[Ollama LLM]
  H --> I[Answer]

🛠️ Технологический стек
Компонент
Инструмент
Эмбеддинги
paraphrase-multilingual-MiniLM-L12-v2 (мультиязычная, ~120 МБ)
Векторная БД
ChromaDB (локальное хранилище)
Генерация
Ollama + qwen2.5:1.5b (сверхлегкая LLM для CPU)
Пайплайн
Чистый Python + sentence-transformers, langchain-text-splitters
Оценка
Rule-based метрики (Faithfulness, Context Relevance, F1)

🚀 Как запустить
Требования
Python 3.9+
Ollama (установлена и запущена)
8 ГБ ОЗУ минимум

Установка

git clone https://github.com/metalltrade1987-star/rag-python-learning.git
cd rag-python-learning
pip install -r requirements.txt
ollama pull qwen2.5:1.5b

1. Индексация базы знаний
python ingest.py

2. Запрос к RAG
python query.py

3. Оценка качества
python evaluate_minimal.py

📁 Структура проекта
├── ingest.py              # Загрузка, чанкинг, векторизация, сохранение в ChromaDB
├── query.py               # Retrieval + Generation пайплайн
├── evaluate_minimal.py    # Rule-based оценка качества RAG
├── knowledge.txt          # Учебный датасет (статья о Python)
└── requirements.txt       # Зависимости

🔍 Инженерные решения и ограничения
Проблема
Решение
8 ГБ ОЗУ
Использованы модели <200 МБ, отключен тяжелый Reranker в оценке, ручное управление памятью (gc.collect())
Англоязычные эмбеддеры/реранкеры на русском тексте
Выявлена проблема на практике, задокументирована необходимость мультиязычных моделей для продакшена
Галлюцинации маленьких LLM
Строгий промпт-инжиниринг + правило отказа при отсутствии контекста