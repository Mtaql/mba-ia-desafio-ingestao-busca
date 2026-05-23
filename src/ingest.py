import hashlib
import json
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import func, select

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

DEFAULT_PROVIDER = "google"


def _get_provider():
    provider = os.getenv("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in {"google", "openai"}:
        raise ValueError(
            "Valor inválido para AI_PROVIDER. Use 'google' ou 'openai'."
        )
    return provider


def _build_embedding_model(provider):
    if provider == "google":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY não configurada para provider 'google'.")
        return GoogleGenerativeAIEmbeddings(model=model)

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY não configurada para provider 'openai'.")
    return OpenAIEmbeddings(model=model)

def extract_text_from_pdf():
    print("Iniciando a extração de texto do PDF...")
    if not PDF_PATH:
        raise ValueError("A variavel de ambiente PDF_PATH nao foi configurada.")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Arquivo PDF nao encontrado em: {PDF_PATH}")

    try:
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        print(f"PDF carregado com sucesso: {PDF_PATH}")
        print(f"Paginas lidas: {len(pages)}")
        return pages
    except Exception as exc:
        raise RuntimeError(f"Falha ao carregar o PDF: {PDF_PATH}") from exc

def split_text(document):
    print("Iniciando a divisão do texto em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(document)
    if not chunks:
        print("Nenhum chunk foi criado. Verifique o conteúdo do documento.")
        raise ValueError("Nenhum chunk foi criado. Verifique o conteúdo do documento.")
    print(f"Texto dividido em {len(chunks)} chunks.")
    return chunks

def enrich_documents(docs):
    print("Iniciando a limpeza e enriquecimento dos documentos...")
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in docs
    ]
    print(f"Documentos enriquecidos: {len(enriched)}")
    return enriched

def build_document_ids(docs):
    ids = []
    for index, doc in enumerate(docs):
        metadata_payload = json.dumps(doc.metadata, sort_keys=True, ensure_ascii=True)
        payload = f"{index}|{metadata_payload}|{doc.page_content}".encode("utf-8")
        ids.append(hashlib.sha256(payload).hexdigest())
    return ids

def count_collection_items(store):
    with store._make_sync_session() as session:
        collection = store.get_collection(session)
        if collection is None:
            return 0

        stmt = select(func.count()).select_from(store.EmbeddingStore).where(
            store.EmbeddingStore.collection_id == collection.uuid
        )
        return session.execute(stmt).scalar_one()

def create_embeddings(embedding_model, docs):
    embeddings = []
    for index, doc in enumerate(docs, start=1):
        embeddings.append(embedding_model.embed_query(doc.page_content))
        print(f"Embedding gerado para o chunk {index}/{len(docs)}")
    return embeddings

def embed_documents(docs):
    print("Iniciando a etapa de embedding dos documentos...")
    provider = _get_provider()
    embedding_model = _build_embedding_model(provider)
    print(f"Provider selecionado para embeddings: {provider}")
    store = PGVector(
        embeddings=embedding_model,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", ""),
        connection=os.getenv("DATABASE_URL", ""),
        use_jsonb=True,
    )
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = build_document_ids(docs)
    embeddings = create_embeddings(embedding_model, docs)
    print("Embedding dos documentos concluído.")
    store.add_embeddings(texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(f"Documentos armazenados no PGVector: {len(docs)}")
    print(f"Total de itens na coleção após ingestão: {count_collection_items(store)}")

def ingest_pdf():
    pages = extract_text_from_pdf()
    chunks = split_text(pages)
    enriched_docs = enrich_documents(chunks)
    embed_documents(enriched_docs)


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as exc:
        print(f"Erro inesperado: {exc}")