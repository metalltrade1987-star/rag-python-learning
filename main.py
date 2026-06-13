import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

# ==========================================
# 1. INGESTION (Загрузка из локального файла)
# ==========================================
print("Читаем локальную базу знаний из knowledge.txt...")
try:
    with open('knowledge.txt', 'r', encoding='utf-8') as file:
        raw_text = file.read()
    print(f"Успешно загружено {len(raw_text)} символов.\n")
except FileNotFoundError:
    print("ОШИБКА: Файл knowledge.txt не найден в папке проекта!")
    exit()

# ==========================================
# 2. CHUNKING (Нарезка на чанки)
# ==========================================
print("Нарезаем текст на чанки...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,      # Чанки поменьше, чтобы модель не запуталась
    chunk_overlap=50,    # Перекрытие
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_text(raw_text)
print(f"Текст разбит на {len(chunks)} чанков.\n")

# ==========================================
# 3. EMBEDDING & RETRIEVAL (Поиск)
# ==========================================
print("Векторизуем чанки...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chunk_embeddings = model.encode(chunks)

# Попробуй разные вопросы!
user_query = "Кто создал Python и в честь чего назвал?"
print(f"\nЗапрос пользователя: '{user_query}'")

query_embedding = model.encode([user_query])[0]

similarities = np.dot(chunk_embeddings, query_embedding) / (
    np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
)

best_match_index = np.argmax(similarities)
retrieved_context = chunks[best_match_index]

print(f"Найден лучший чанк (сходство: {similarities[best_match_index]:.4f}):")
print(f"-> {retrieved_context}\n")

# ==========================================
# 4. GENERATION (Генерация ответа)
# ==========================================
prompt = f"""Ты эксперт по Python. Ответь на вопрос, основываясь ИСКЛЮЧИТЕЛЬНО на этом тексте:
Текст: {retrieved_context}
Вопрос: {user_query}
Правило: Дай краткий и точный ответ. Если в тексте нет информации для ответа, так и скажи.
Ответ:"""

print("Генерируем ответ через Ollama...")
try:
    response = ollama.chat(
        model='qwen2.5:1.5b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1, 'num_predict': 150}
    )
    print("\n" + "="*50)
    print("ОТВЕТ НЕЙРОСЕТИ (на основе твоего файла):")
    print("="*50)
    print(response['message']['content'])
except Exception as e:
    print(f"\nОшибка Ollama: {e}")