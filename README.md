# 🤖 LINA IA — Intelligent Assistant Framework

**LINA IA** é uma plataforma modular para o desenvolvimento de **assistentes conversacionais inteligentes**, criada por estudantes da **Universidade Federal Fluminense (UFF)**.  
O projeto combina **LLMs**, **memória conversacional**, e **busca vetorial (RAG)** para oferecer experiências personalizadas e seguras — voltada ao apoio de pessoas com diabetes.

---

## 🚀 Visão Geral

LINA IA permite criar agentes inteligentes com:

- **LLMs configuráveis** (OpenAI GPT-4/GPT-3.5)
- **RAG com Qdrant** para busca semântica eficiente
- **Memória persistente via Redis**
- **Prompts de sistema customizáveis, utilizando o Framework LLAMAINDEX**
- **Ferramentas (Tools)** que ampliam as capacidades dos agentes
- **Suporte a múltiplos usuários**, com isolamento de contexto

---

## ⚙️ Como Executar o Projeto

Siga o passo a passo abaixo para configurar e rodar o assistente conversacional localmente utilizando um ambiente virtual Python (`venv`).

### 1. Pré-requisitos
- **Python 3.10+** instalado
- **Docker** e **Docker Compose** instalados (para subir o Qdrant e o Redis)
- Chave de API da **OpenAI**

### 2. Configurar o Ambiente Virtual (venv)
É altamente recomendado isolar as dependências do projeto num ambiente virtual para evitar conflitos de bibliotecas.

```bash
# Navegue até a pasta do projeto
cd Lina-IA

# Crie o ambiente virtual
python3 -m venv venv

# Ative o ambiente virtual
# No Linux/MacOS/WSL:
source venv/bin/activate
# No Windows (PowerShell):
 .\venv\Scripts\Activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente
O projeto precisa de chaves de API e caminhos que ficam no arquivo `.env`.

```bash
# Copie o arquivo de template
cp .env.template .env
```
Abra o `.env` no seu editor de código e cole a sua `OPEN_API_KEY`.

### 4. Subir a Infraestrutura Base (Docker)
A memória do chat e o banco vetorial rodam em containers via Docker Compose.

```bash
# Subir os containers em background
docker compose up -d
```

> **Nota:** O dashboard do Qdrant pode ser acessado no navegador através de [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### 5. Ingestão de Dados (Obrigatório na 1ª Vez)
Como o banco vetorial inicializa vazio, você precisa preenchê-lo e criar a coleção `lina_docs_tb`. Coloque seus arquivos `.pdf` para formar a base de conhecimento (RAG) na pasta configurada em `TB_DOCS_PATH` (por padrão, `data/raw/`) e rode a indexação:

```bash
./scripts/ingest-tb.sh
```

### 6. Iniciar a Interação (Chat)
Com os containers ativos e o ambiente configurado, rode o script do chat engine para conversar com a Lina diretamente no terminal:

```bash
./scripts/chatengine-tb.sh
```

*(Para fazer consultas rápidas no banco vetorial sem o histórico de chat, você também pode usar `./scripts/queryengine.sh "Sua Pergunta"`).*
