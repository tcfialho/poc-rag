#!/usr/bin/env python3
"""
Pipeline RAG on-premises
------------------------
• Cria ou carrega um índice FAISS persistente  
• Indexa um .txt/.md em chunks + embeddings (Sentence-Transformers)  
• Usa um PromptNode (OpenAI, Claude ou Gemini) para gerar as respostas  
• Mantém histórico curto de conversa entre perguntas

Estrutura principal
1. Parser / paths e constantes                          
2. Carregar ou criar o FAISSDocumentStore               
3. Instanciar o EmbeddingRetriever    
4. Atualizar embeddings se índice for novo              
5. Construir o pipeline (Retriever → Generator)         
6. Loop de interação com histórico                      
"""

import argparse
import os
import uuid
from typing import List, Dict, Tuple

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.pipelines import Pipeline
from haystack.schema import Document

# Constantes de configuração
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
CHUNK_SIZE      = 500
SQL_URL         = "sqlite:///faiss_document_store.db"


class ConversationTracker:
    """Mantém as últimas *n* interações usuário-assistente."""
    def __init__(self, max_history: int = 5) -> None:
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history
        self.session_id  = str(uuid.uuid4())

    def add(self, query: str, response: str) -> None:
        self.history.append({"query": query, "response": response})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def formatted_history(self) -> str:
        if not self.history:
            return ""
        return "\n".join(
            f"Pergunta anterior: {h['query']}\nResposta anterior: {h['response']}"
            for h in self.history
        )


def get_prompt_node(provider: str) -> PromptNode:
    """Factory simples para PromptNodes (OpenAI, Claude ou Gemini)."""
    tmpl = """
Com base no contexto e no histórico abaixo, responda à pergunta atual.
Se faltar informação, diga que não há dados suficientes.
Contexto: {documents}

{query}

Resposta:"""

    provider = provider.lower()
    key_env  = {
        "openai": "OPENAI_API_KEY",
        "claude": "CLAUDE_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }.get(provider)

    if not key_env:
        raise ValueError(f"Provedor '{provider}' não suportado.")

    return PromptNode(
        model_name_or_path={"openai": "gpt-4.1-nano",
                            "claude": "claude",
                            "gemini": "gemini"}[provider],
        api_key=os.getenv(key_env),
        default_prompt_template=tmpl,
        use_gpu=False,
    )


def load_or_create_document_store(doc_path: str, persist_dir: str
                                  ) -> Tuple[FAISSDocumentStore, bool]:
    """Carrega índice se existir; senão cria, indexa e retorna flag `novo`."""
    os.makedirs(persist_dir, exist_ok=True)
    index_path  = os.path.join(persist_dir, "faiss_index.faiss")
    config_path = os.path.join(persist_dir, "faiss_index.json")

    if os.path.exists(index_path) and os.path.exists(config_path):
        try:
            store = FAISSDocumentStore.load(index_path=index_path,
                                            config_path=config_path)
            return store, False
        except ValueError:
            # ín-matching dimensional ou corrupção → recria
            for f in (index_path, config_path, SQL_URL.split("///")[-1]):
                if os.path.exists(f):
                    os.remove(f)

    # --- Criar novo índice ---
    store = FAISSDocumentStore(
        faiss_index_factory_str="Flat",
        embedding_dim=EMBEDDING_DIM,
        sql_url=SQL_URL,
        return_embedding=True,
        duplicate_documents="skip",
        similarity="dot_product"
    )

    # Lê arquivo e quebra em chunks
    with open(doc_path, encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i: i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    docs   = [Document(content=c,
                       meta={"source": doc_path, "name": f"chunk_{i}"})
              for i, c in enumerate(chunks)]
    store.write_documents(docs)
    return store, True


def create_retriever(store: FAISSDocumentStore) -> EmbeddingRetriever:
    """Retorna o EmbeddingRetriever (instanciado apenas uma vez)."""
    return EmbeddingRetriever(
        document_store=store,
        embedding_model=EMBEDDING_MODEL,
        use_gpu=False,
        top_k=1
    )

def update_and_save_index(store: FAISSDocumentStore,
                          retriever: EmbeddingRetriever,
                          persist_dir: str) -> None:
    """Gera/atualiza embeddings e persiste índice FAISS + json config."""
    index_path  = os.path.join(persist_dir, "faiss_index.faiss")
    config_path = os.path.join(persist_dir, "faiss_index.json")
    store.update_embeddings(retriever, update_existing_embeddings=True)
    store.save(index_path=index_path, config_path=config_path)

def build_pipeline(retriever: EmbeddingRetriever,
                   prompt_node: PromptNode) -> Pipeline:
    pipe = Pipeline()
    pipe.add_node(retriever, name="Retriever", inputs=["Query"])
    pipe.add_node(prompt_node, name=" or", inputs=["Retriever"])
    return pipe

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline RAG simples usando FAISS + Sentence-Transformers."
    )
    parser.add_argument("--provider", type=str, required=True,
                        choices=["openai", "claude", "gemini"],
                        help="LLM backend (openai, claude, gemini)")
    parser.add_argument("--doc_path", type=str, required=True,
                        help="Caminho para arquivo .txt ou .md a indexar")
    args = parser.parse_args()

    if not (os.path.isfile(args.doc_path)
            and args.doc_path.lower().endswith((".txt", ".md"))):
        raise ValueError("doc_path deve apontar para arquivo .txt ou .md existente.")

    persist_dir = "faiss_index"

    # 2. Carregar ou criar índice
    document_store, novo_indice = load_or_create_document_store(args.doc_path,
                                                                persist_dir)

    # 3. Retriever único
    retriever = create_retriever(document_store)

    # 4. Atualiza embeddings se índice recém-criado
    if novo_indice:
        update_and_save_index(document_store, retriever, persist_dir)

    # 5. Pipeline + PromptNode
    prompt_node  = get_prompt_node(args.provider)
    pipe         = build_pipeline(retriever, prompt_node)
    conversation = ConversationTracker(max_history=5)

    print(f"ID da Sessão: {conversation.session_id}")
    print("Digite sua pergunta (ou 'sair' para encerrar).")

    # 6. Loop de interação
    while True:
        user_q = input("\nPergunta: ").strip()
        if user_q.lower() == "sair":
            break

        history   = conversation.formatted_history()
        full_q    = (f"Histórico da conversa:\n{history}\n\nPergunta atual: {user_q}"
                     if history else user_q)

        result    = pipe.run(query=full_q)
        answer    = result.get("results", ["Não foi possível gerar resposta."])[0]

        print("\nResposta:", answer)
        conversation.add(user_q, answer)


if __name__ == "__main__":
    main()
