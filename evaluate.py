import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ==========================================
# 1. НАБОР ТЕСТОВЫХ ВОПРОСОВ (Ground Truth)
# ==========================================
# Это "экзамен" для нашего RAG. Мы знаем правильные ответы и контексты.
test_data = {
    "question": [
        "Кто создал Python?",
        "В каком году появился Python?",
        "В честь чего назван Python?",
        "Какое ключевое слово используется для создания функции?",
        "Чем кортежи отличаются от списков?",
        "Кто такой Илон Маск?"  # Вопрос-ловушка
    ],
    "ground_truth": [
        "Python создал голландский разработчик Гвидо ван Россум.",
        "Первая версия Python была выпущена в 1991 году.",
        "Python назван в честь британского комедийного шоу 'Летающий цирк Монти Пайтона'.",
        "Для создания функции используется ключевое слово def.",
        "Списки являются изменяемыми последовательностями, а кортежи — неизменяемыми.",
        "В предоставленной базе знаний нет информации об Илоне Маске."
    ]
}

# ==========================================
# 2. ФУНКЦИЯ: Прогнать вопрос через наш RAG
# ==========================================
print("Загружаем базу данных и модели...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="python_knowledge")
retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def ask_rag(question: str):
    """Прогоняет вопрос через наш RAG и возвращает контекст + ответ."""
    # Поиск
    query_embedding = retriever_model.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    candidates = results['documents'][0]

    # Реранкинг
    pairs = [[question, c] for c in candidates]
    scores = reranker_model.predict(pairs)
    best_idx = scores.argmax()
    context = candidates[best_idx]

    # Генерация
    prompt = f"""Ты эксперт по Python. Ответь на вопрос, основываясь ИСКЛЮЧИТЕЛЬНО на этом тексте:
Текст: {context}
Вопрос: {question}
Правило: Дай краткий ответ. Если в тексте нет информации, так и скажи.
Ответ:"""

    response = ollama.chat(
        model='qwen2.5:1.5b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1, 'num_predict': 150}
    )
    answer = response['message']['content']

    return context, answer


# ==========================================
# 3. ПРОГОНЯЕМ ВСЕ ВОПРОСЫ ЧЕРЕЗ RAG
# ==========================================
print("\nПрогоняем вопросы через RAG (это займет 1-2 минуты)...")
contexts_list = []
answers_list = []

for i, q in enumerate(test_data["question"]):
    print(f"  [{i + 1}/{len(test_data['question'])}] {q}")
    ctx, ans = ask_rag(q)
    contexts_list.append([ctx])  # RAGAS ждет список контекстов
    answers_list.append(ans)
    print(f"       Ответ: {ans[:80]}...")

# Добавляем ответы и контексты в набор данных
test_data["answer"] = answers_list
test_data["contexts"] = contexts_list

# ==========================================
# 4. ОЦЕНКА ЧЕРЕЗ RAGAS
# ==========================================
print("\n🔍 RAGAS оценивает качество (это займет еще пару минут)...")
print("   RAGAS использует LLM как экзаменатора для проверки ответов.\n")

dataset = Dataset.from_dict(test_data)

# ВАЖНО: RAGAS нужна LLM для оценки. Настраиваем на локальную Ollama.
# Для этого мы используем обертку Langchain.
try:
    from langchain_community.llms import Ollama
    from ragas.llms import LangchainLLMWrapper

    # Настраиваем "экзаменатора" на локальную модель
    evaluator_llm = LangchainLLMWrapper(Ollama(model="qwen2.5:1.5b"))

    # Запускаем оценку
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm
    )

    # ==========================================
    # 5. ВЫВОД РЕЗУЛЬТАТОВ
    # ==========================================
    print("=" * 60)
    print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ RAG")
    print("=" * 60)
    print(results)
    print("\n")
    print("📈 РАСШИФРОВКА:")
    print(f"  Faithfulness (Верность):      {results['faithfulness']:.3f}  — ответ следует из контекста?")
    print(f"  Answer Relevancy (Релевант.): {results['answer_relevancy']:.3f}  — ответ по делу?")
    print(f"  Context Precision (Точность): {results['context_precision']:.3f}  — поиск нашел нужное?")
    print(f"  Context Recall (Полнота):     {results['context_recall']:.3f}  — контекста хватило?")
    print("\n💡 Идеальное значение: 1.0. Чем ближе, тем лучше.")

except Exception as e:
    print(f"\n❌ Ошибка при оценке: {e}")
    print("\nЕсли RAGAS не смог запуститься с локальной моделью,")
    print("мы можем посчитать простые метрики вручную — без 'экзаменатора-LLM'.")