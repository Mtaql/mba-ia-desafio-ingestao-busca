# Desafio MBA Engenharia de Software com IA - Full Cycle

## Executando com Docker (Python 3.14.4)

Esta configuração garante que qualquer pessoa rode os scripts com a mesma versão de Python (`3.14.4`) e com Postgres + pgvector prontos.

### 1) Pré-requisitos

- Docker instalado
- Docker Compose disponível (`docker compose`)

### 2) Configure as variáveis de ambiente

Antes de subir os containers, copie o arquivo de exemplo e preencha as chaves:

```bash
cp .env.example .env
```

Em seguida, edite o arquivo `.env` e informe os valores reais (principalmente API keys e nome da coleção no banco).

Campos obrigatórios em `.env`:

- `AI_PROVIDER` (`google` ou `openai`)
- `PG_VECTOR_COLLECTION_NAME`

Campos obrigatórios por provider:

- Se `AI_PROVIDER=google`:
	- `GOOGLE_API_KEY`
	- `GOOGLE_EMBEDDING_MODEL` (opcional, padrão: `gemini-embedding-2-preview`)
	- `GOOGLE_CHAT_MODEL` (opcional, padrão: `gemini-2.5-flash`)
- Se `AI_PROVIDER=openai`:
	- `OPENAI_API_KEY`
	- `OPENAI_EMBEDDING_MODEL` (opcional, padrão: `text-embedding-3-small`)
	- `OPENAI_CHAT_MODEL` (opcional, padrão: `gpt-4o-mini`)

Exemplos de configuração:

```env
# Google
AI_PROVIDER=google
GOOGLE_API_KEY=...
GOOGLE_EMBEDDING_MODEL=gemini-embedding-2-preview
GOOGLE_CHAT_MODEL=gemini-2.5-flash
```

```env
# OpenAI
AI_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

Momento em que o `.env` é utilizado:

- Ao executar `docker compose up -d --build`, o Compose lê o `.env` local e injeta as variáveis no container.
- Se você alterar o `.env` depois que os containers já estiverem rodando, recrie o serviço para aplicar os novos valores.

Para reaplicar variáveis após alteração no `.env`:

```bash
docker compose up -d --force-recreate app
```

Observações importantes:

- Dentro do container, o banco usa `postgres` como host (isso já é forçado no `docker-compose.yml`).
- O PDF no container aponta para `/app/document.pdf` por padrão. Para alterar, defina `PDF_PATH_CONTAINER` no `.env`.

### 3) Suba os serviços

```bash
docker compose up -d --build
```

Verifique a versão do Python no container:

```bash
docker compose exec app python --version
```

Saída esperada:

```text
Python 3.14.4
```

### 4) Rode os scripts

Ingestão do PDF:

```bash
docker compose exec app python src/ingest.py
```

Busca de exemplo:

```bash
docker compose exec app python src/search.py
```

Chat interativo:

```bash
docker compose exec app python src/chat.py
```

### 5) Encerrar ambiente

```bash
docker compose down
```

Para remover também o volume do banco:

```bash
docker compose down -v
```