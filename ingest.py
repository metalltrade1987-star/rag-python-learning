import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("1. Читаем базу знаний...")
with open('knowledge.txt', 'r', encoding='utf-8') as file:
    raw_text = file.read()

print("2. Нарезаем текст на чанки...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_text(raw_text)
print(f"   Получено {len(chunks)} чанков.")

print("3. Векторизуем чанки (это займет время, но только один раз!)...")
# Та же мультиязычная модель
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chunk_embeddings = model.encode(chunks)

print("4. Сохраняем в векторную базу данных ChromaDB...")
# Создаем постоянную базу данных в папке ./chroma_db
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Создаем или получаем коллекцию (как таблицу в обычной БД)
collection = chroma_client.get_or_create_collection(name="python_knowledge")

# Генерируем уникальные ID для каждого чанка
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Добавляем документы, их векторы и ID в базу
collection.add(
    documents=chunks,
    embeddings=chunk_embeddings.tolist(), # Chroma требует списки, а не numpy массивы
    ids=ids
)

print("✅ ИНДЕКСАЦИЯ ЗАВЕРШЕНА! База сохранена в папку 'chroma_db'.")
print("Теперь этот скрипт можно не запускать, пока не изменится knowledge.txt")