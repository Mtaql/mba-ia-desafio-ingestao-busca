import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

DEFAULT_PROVIDER = "google"


def _get_provider():
    provider = os.getenv("AI_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in {"google", "openai"}:
        raise ValueError(
            "Valor inválido para AI_PROVIDER. Use 'google' ou 'openai'."
        )
    return provider


def _build_embeddings(provider):
    if provider == "google":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2-preview")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY não configurada para provider 'google'.")
        return GoogleGenerativeAIEmbeddings(model=model)

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY não configurada para provider 'openai'.")
    return OpenAIEmbeddings(model=model)


def _build_chat_model(provider):
    if provider == "google":
        model_name = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY não configurada para provider 'google'.")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)

    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY não configurada para provider 'openai'.")
    return ChatOpenAI(model=model_name, temperature=0)

def _build_vector_store():
    provider = _get_provider()
    embeddings = _build_embeddings(provider)
    print(f"Provider selecionado para busca: {provider}")
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", ""),
        connection=os.getenv("DATABASE_URL", ""),
        use_jsonb=True,
    )


def _format_context(results):
    context_blocks = []
    for index, (doc, score) in enumerate(results, start=1):
        metadata = ", ".join(
            f"{key}={value}" for key, value in sorted(doc.metadata.items())
        ) or "sem metadados"
        context_blocks.append(
            f"Trecho {index} (score: {score:.4f}; {metadata})\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_blocks)


def search_prompt(question=None, k=10, verbose=True):
    print("Iniciando a etapa de busca com base na pergunta do usuário...")
    if not question:
        print("Nenhuma pergunta fornecida para a busca.")
        return None

    store = _build_vector_store()
    results = store.similarity_search_with_score(question, k=k)
    print(f"Busca concluída. Número de resultados encontrados: {len(results)} verbose {verbose}")

    if verbose:
        for i, (doc, score) in enumerate(results, start=1):
            print("=" * 50)
            print(f"Resultado {i} (score: {score:.2f}):")
            print("=" * 50)

            print("\nTexto:\n")
            print(doc.page_content.strip())

            print("\nMetadados:\n")
            for key, value in doc.metadata.items():
                print(f"{key}: {value}")

    return results


def build_answer_chain():
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    provider = _get_provider()
    llm = _build_chat_model(provider)
    return prompt | llm | StrOutputParser()


def answer_question(question, k=10, verbose=False):
    if not question:
        print("Nenhuma pergunta fornecida para o chat.")
        return None

    results = search_prompt(question, k=k, verbose=verbose)
    if not results:
        return "Não tenho informações necessárias para responder sua pergunta."

    context = _format_context(results)
    chain = build_answer_chain()
    return chain.invoke({"contexto": context, "pergunta": question})

if __name__ == "__main__":
    try:
        response = answer_question(
            question="qual o faturamento de Mirage Entretenimento Participações?",
            k=10,
            verbose=True
        )
        print("\nResposta:\n")
        print(response)
    except Exception as exc:
        print(f"Erro durante a busca: {exc}")