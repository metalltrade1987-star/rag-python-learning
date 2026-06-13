import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# ==========================================
# 1. ЗАГРУЗКА БАЗЫ И МОДЕЛЕЙ
# ==========================================
print("1. Загружаем базу данных и модели...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="python_knowledge")

# Модель для быстрого поиска (Поисковик)
retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Модель-Судья (Reranker). Она тяжелее, но точнее.
# Используем легкую версию, чтобы твой Ryzen не тормозил.
print("   (Загружаем модель-судью, это займет 10-20 сек при первом запуске...)")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# 2. ПОИСК (Retrieval) - Ищем ТОП-5 кандидатов
# ==========================================
user_query = "Who is Ilon Mask?"
print(f"\nВопрос: '{user_query}'")

print("2. Ищем 5 кандидатов в базе...")
query_embedding = retriever_model.encode([user_query])[0].tolist()

# Просим Chroma вернуть сразу 5 лучших чанков
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

candidates = results['documents'][0]
print(f"   Поисковик нашел {len(candidates)} кандидатов.")

# ==========================================
# 3. СУДЬЯ (Reranking) - Выбираем лучшего из лучших
# ==========================================
print("3. Судья перепроверяет кандидатов...")

# Создаем пары: [Вопрос, Кандидат 1], [Вопрос, Кандидат 2] и т.д.
pairs = [[user_query, candidate] for candidate in candidates]

# Судья ставит каждой паре оценку (score)
scores = reranker_model.predict(pairs)

# Находим индекс чанка с максимальной оценкой от Судьи
best_index = scores.argmax()
best_score = scores[best_index]
retrieved_context = candidates[best_index]

print(f"   Лучший чанк (оценка судьи: {best_score:.4f}):")
print(f"   -> {retrieved_context[:100]}...\n")

# ==========================================
# 4. ГЕНЕРАЦИЯ (Generation)
# ==========================================
print("4. Генерируем ответ через Ollama...")
prompt = f"""Ты эксперт по Python. Ответь на вопрос, основываясь ИСКЛЮЧИТЕЛЬНО на этом тексте:
Текст: {retrieved_context}
Вопрос: {user_query}
Правило: Дай краткий и точный ответ. Если в тексте нет информации для ответа, так и скажи.
Ответ:"""

try:
    response = ollama.chat(
        model='qwen2.5:1.5b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1, 'num_predict': 150}
    )
    print("="*50)
    print("ОТВЕТ НЕЙРОСЕТИ:")
    print("="*50)
    print(response['message']['content'])
except Exception as e:
    print(f"Ошибка Ollama: {e}")