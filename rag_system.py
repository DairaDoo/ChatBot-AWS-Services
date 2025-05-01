import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
from typing import List, Tuple


class RAGSystem:
    def __init__(self, knowledge_base_path: str, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Inicializa el sistema RAG.
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.documents = self._load_and_chunk_documents(knowledge_base_path)
        self.embeddings = self._create_embeddings()
        self._create_faiss_index()

        print(f"Sistema RAG inicializado con {len(self.documents)} fragmentos")

    def _load_and_chunk_documents(self, file_path: str) -> List[str]:
        if not os.path.exists(file_path):
            print(f"Archivo no encontrado: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.chunk_size:
                chunks.append(paragraph)
            else:
                start = 0
                while start < len(paragraph):
                    end = start + self.chunk_size
                    chunks.append(paragraph[start:end].strip())
                    start += (self.chunk_size - self.chunk_overlap)

        return chunks

    def _create_embeddings(self) -> np.ndarray:
        return self.model.encode(self.documents)

    def _create_faiss_index(self):
        vector_dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                similarity = 1.0 / (1.0 + distances[0][i])
                results.append((self.documents[idx], similarity))

        return results

    def _normalize_query(self, query: str) -> str:
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)
        return query

    def _extract_keywords(self, query: str) -> List[str]:
        stopwords = {
            "que", "es", "aws", "me", "puedes", "decir", "dame", "información",
            "sobre", "de", "los", "las", "un", "una", "para", "el", "la"
        }
        normalized = self._normalize_query(query)
        words = normalized.split()
        return [word for word in words if word not in stopwords]

    def generate_response(self, query: str) -> str:
        keywords = self._extract_keywords(query)
        if not keywords:
            return "Por favor, intenta reformular tu pregunta."

        search_query = " ".join(keywords)
        retrieved_docs = self.retrieve(search_query)

        if not retrieved_docs:
            return "Lo siento, no tengo información sobre eso."

        for doc, _ in retrieved_docs:
            lines = doc.splitlines()
            for line in lines:
                if any(kw in line.lower() for kw in keywords):
                    return line.strip()

        return "No encontré información suficientemente relevante para responder tu pregunta."
