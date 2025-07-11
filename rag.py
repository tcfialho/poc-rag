#!/usr/bin/env python3
"""
Pipeline RAG on-premises com Haystack 2.0
--------------------------------------------
• Usa ChromaDocumentStore para armazenamento vetorial persistente.
• Indexa um .txt/.md em chunks + embeddings (Sentence-Transformers).
• Usa OpenRouterChatGenerator para gerar as respostas.
• Mantém histórico curto de conversa entre perguntas.

Estrutura principal:
1.  Paths e constantes.
2.  Inicialização do Document Store e do Embedder.
3.  Lógica para indexar o documento (se necessário).
4.  Construção do pipeline RAG com componentes.
5.  Loop de interação com histórico.
"""
import os
import argparse
import uuid
import shutil
from pathlib import Path
from typing import List, Dict
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.dataclasses import ChatMessage, Document
from haystack.utils import Secret
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator
import glob

# Constantes de configuração
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHROMA_PATH = "./chroma_db"

class ConversationTracker:
    """Mantém as últimas *n* interações usuário-assistente."""
    def __init__(self, max_history: int = 5) -> None:
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.session_id = str(uuid.uuid4())

    def add(self, query: str, response: str) -> None:
        self.history.append({"query": query, "response": response})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def formatted_history(self) -> str:
        if not self.history:
            return "Nenhum histórico de conversa ainda."
        return "\n".join(
            f"Pergunta anterior: {h['query']}\nResposta anterior: {h['response']}"
            for h in self.history
        )

def index_document_if_needed(document_store: ChromaDocumentStore, rebuild: bool):
    """Verifica se o DocumentStore está vazio e o indexa se necessário, ou força rebuild se especificado, ingerindo arquivos de 'ks/'."""
    ks_dir = "ks"

    if rebuild or document_store.count_documents() == 0:
        if not os.path.exists(ks_dir):
            print(f"Pasta '{ks_dir}' não existe. Crie-a e adicione arquivos .txt ou .md.")
            return

        file_paths = glob.glob(os.path.join(ks_dir, '*.txt')) + glob.glob(os.path.join(ks_dir, '*.md'))
        if not file_paths:
            print(f"Nenhum arquivo .txt ou .md encontrado em '{ks_dir}'. Pulando a indexação.")
            return

        print(f"Indexando {len(file_paths)} arquivos de '{ks_dir}'...")
        all_docs = []
        for file_path in file_paths:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
            chunks = [text[i: i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            docs = [Document(content=c, meta={"source": file_path}) for c in chunks]
            all_docs.extend(docs)

        embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
        embedder.warm_up()
        embedded_docs = embedder.run(documents=all_docs)["documents"]

        document_store.write_documents(embedded_docs, policy=DuplicatePolicy.OVERWRITE)
        print(f"Documentos indexados: {document_store.count_documents()}")
    else:
        print(f"DocumentStore já contém {document_store.count_documents()} documentos. Pulando a indexação.")

def build_rag_pipeline(document_store: ChromaDocumentStore, model_name: str) -> Pipeline:
    prompt_template = [ChatMessage.from_system("""
    Com base no contexto dos documentos e no histórico da conversa abaixo, responda à pergunta atual.
    Se a informação não estiver no contexto, diga que não há dados suficientes para responder.

    Contexto dos documentos:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    Histórico da conversa:
    {{ history }}

    Pergunta atual: {{ query }}
    Resposta:
    """)]

    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
    retriever = ChromaEmbeddingRetriever(document_store=document_store, top_k=2)
    prompt_builder = ChatPromptBuilder(template=prompt_template, required_variables=["documents", "history", "query"])
    llm = OpenRouterChatGenerator(model=model_name)

    rag_pipeline = Pipeline()
    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", llm)

    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    return rag_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline RAG com Haystack 2.0, Chroma e OpenRouter.")
    parser.add_argument("--model", type=str, default="deepseek/deepseek-chat", help="Nome do modelo OpenRouter a ser usado.")
    parser.add_argument("--rebuild-index", action="store_true", help="Força a recriação do índice de documentos, deletando e reindexando.")
    args = parser.parse_args()

    if args.rebuild_index and os.path.exists(CHROMA_PATH):
        print("Limpando o banco de dados Chroma existente para recriação...")
        shutil.rmtree(CHROMA_PATH)

    document_store = ChromaDocumentStore(persist_path=CHROMA_PATH)
    index_document_if_needed(document_store, args.rebuild_index)

    rag_pipeline = build_rag_pipeline(document_store, args.model)
    conversation = ConversationTracker(max_history=3)

    print("-" * 50)
    print(f"ID da Sessão: {conversation.session_id}")
    print("Pipeline RAG pronto. Digite sua pergunta ou 'sair' para encerrar.")
    print("-" * 50)

    while True:
        user_q = input("\nPergunta: ").strip()
        if user_q.lower() == "sair":
            break

        history = conversation.formatted_history()

        result = rag_pipeline.run(data={
            "text_embedder": {"text": user_q},
            "prompt_builder": {"query": user_q, "history": history}
        })

        answer = result["llm"]["replies"][0].text
        print("\nResposta:", answer)
        conversation.add(user_q, answer)

if __name__ == "__main__":
    if "OPENROUTER_API_KEY" not in os.environ:
        raise ValueError("Por favor, defina a variável de ambiente OPENROUTER_API_KEY.")
    main()
