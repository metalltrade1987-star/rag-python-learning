import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama
import re

# ==========================================
# 1. ТЕСТОВЫЕ ВОПРОСЫ С ЭТАЛОННЫМИ ОТВЕТАМИ
# ==========================================
test_data = [
    {
        "question": "Кто создал Python?",
        "ground_truth": "Python создал голландский разработчик Гвидо ван Россум."
    },
    {
        "question": "В каком году появился Python?",
        "ground_truth": "Первая версия Python была выпущена в 1991 году."
    },
    {
        "question": "В честь чего назван Python?",
        "ground_truth": "Python назван в честь британского комедийного шоу Летающий цирк Монти Пайтона."
    },
    {
        "question": "Какое ключевое слово используется для создания функции?",
        "ground_truth": "Для создания функции используется ключевое слово def."
    },
    {
        "question": "Чем кортежи отличаются от списков?",
        "ground_truth": "Списки являются изменяемыми последовательностями, а кортежи — неизменяемыми."
    },
    {
        "question": "Кто такой Илон Маск?",
        "ground_truth": "В предоставленной базе знаний нет информации об Илоне Маске."
    }
]

# ==========================================
# 2. ЗАГРУЗКА НАШЕГО RAG
# ==========================================
print("Загружаем базу данных и модели...")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="python_knowledge")
retriever_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def ask_rag(question: str):
    """Прогоняет вопрос через наш RAG."""
    query_embedding = retriever_model.encode([question])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    candidates = results['documents'][0]

    pairs = [[question, c] for c in candidates]
    scores = reranker_model.predict(pairs)
    best_idx = scores.argmax()
    context = candidates[best_idx]

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
    return context, response['message']['content']


# ==========================================
# 3. ФУНКЦИИ ДЛЯ ПОДСЧЕТА МЕТРИК
# ==========================================
def tokenize(text):
    """Разбивает текст на слова, приводя к нижнему регистру."""
    # Убираем пунктуацию и разбиваем по пробелам
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return set(text.split())


def faithfulness_proxy(answer, context):
    """Доля слов ответа, которые есть в контексте."""
    ans_words = tokenize(answer)
    ctx_words = tokenize(context)
    if not ans_words:
        return 0.0
    overlap = ans_words & ctx_words
    return len(overlap) / len(ans_words)


def context_relevance(question, context):
    """Доля слов вопроса, которые есть в контексте."""
    q_words = tokenize(question)
    ctx_words = tokenize(context)
    if not q_words:
        return 0.0
    overlap = q_words & ctx_words
    return len(overlap) / len(q_words)


def answer_completeness(answer):
    """Нормализованная длина ответа (идеал: 10-30 слов)."""
    words = len(tokenize(answer))
    if words == 0:
        return 0.0
    if words < 5:
        return 0.3  # Слишком короткий
    if words > 50:
        return 0.7  # Слишком длинный
    return 1.0  # Нормальная длина


def f1_score(prediction, reference):
    """Классическая F1-мера между ответом и эталоном."""
    pred_words = tokenize(prediction)
    ref_words = tokenize(reference)
    if not pred_words or not ref_words:
        return 0.0
    overlap = pred_words & ref_words
    if not overlap:
        return 0.0
    precision = len(overlap) / len(pred_words)
    recall = len(overlap) / len(ref_words)
    return 2 * (precision * recall) / (precision + recall)


# ==========================================
# 4. ПРОГОНЯЕМ ТЕСТЫ И СЧИТАЕМ МЕТРИКИ
# ==========================================
print("\nПрогоняем вопросы через RAG...\n")
metrics = {
    "faithfulness": [],
    "context_relevance": [],
    "completeness": [],
    "f1_vs_truth": []
}

for i, item in enumerate(test_data):
    q = item["question"]
    gt = item["ground_truth"]
    print(f"[{i + 1}/{len(test_data)}] Вопрос: {q}")

    context, answer = ask_rag(q)

    f = faithfulness_proxy(answer, context)
    cr = context_relevance(q, context)
    comp = answer_completeness(answer)
    f1 = f1_score(answer, gt)

    metrics["faithfulness"].append(f)
    metrics["context_relevance"].append(cr)
    metrics["completeness"].append(comp)
    metrics["f1_vs_truth"].append(f1)

    print(f"   Ответ RAG: {answer[:80]}...")
    print(f"   Метрики: Faith={f:.2f} | CtxRel={cr:.2f} | Compl={comp:.2f} | F1={f1:.2f}\n")

# ==========================================
# 5. ИТОГОВАЯ ТАБЛИЦА
# ==========================================
print("=" * 60)
print("📊 ИТОГОВЫЕ МЕТРИКИ КАЧЕСТВА RAG")
print("=" * 60)

avg_faith = sum(metrics["faithfulness"]) / len(metrics["faithfulness"])
avg_cr = sum(metrics["context_relevance"]) / len(metrics["context_relevance"])
avg_comp = sum(metrics["completeness"]) / len(metrics["completeness"])
avg_f1 = sum(metrics["f1_vs_truth"]) / len(metrics["f1_vs_truth"])

print(f"  Faithfulness (Верность контексту):  {avg_faith:.3f}  — модель не выдумывает?")
print(f"  Context Relevance (Релев. контекст): {avg_cr:.3f}  — поиск находит нужное?")
print(f"  Completeness (Полнота ответа):       {avg_comp:.3f}  — ответы не слишком короткие?")
print(f"  F1 vs Ground Truth (Точность):       {avg_f1:.3f}  — совпадение с эталоном?")
print("\n💡 Идеальное значение: 1.0")
print("=" * 60)

# ==========================================
# 6. АНАЛИЗ СЛАБЫХ МЕСТ
# ==========================================
print("\n🔍 АНАЛИЗ ПО ВОПРОСАМ:")
for i, item in enumerate(test_data):
    q = item["question"]
    f = metrics["faithfulness"][i]
    cr = metrics["context_relevance"][i]
    f1 = metrics["f1_vs_truth"][i]

    problems = []
    if f < 0.5:
        problems.append("⚠️ модель выдумывает")
    if cr < 0.2:
        problems.append("⚠️ поиск нашел не то")
    if f1 < 0.3:
        problems.append("⚠️ ответ далек от эталона")

    status = "✅ OK" if not problems else " | ".join(problems)
    print(f"  [{i + 1}] {q[:50]:<50} → {status}")