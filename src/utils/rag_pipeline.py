"""RAG pipeline wrapper supporting multiple architectures."""
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    doc_id: str
    text: str
    score: float
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResult:
    query: str
    passages: List[Passage]
    output: str
    certificate: Optional[Dict] = None
    flagged_passages: List[Passage] = field(default_factory=list)
    verified_passages: List[Passage] = field(default_factory=list)
    overhead_ms: float = 0.0


class BM25Retriever:
    def __init__(self, corpus_path: str):
        self.corpus = self._load_corpus(corpus_path)
        self.index = None
        self._build_index()

    def _load_corpus(self, path: str) -> List[Dict]:
        import jsonlines
        docs = []
        if os.path.exists(path):
            with jsonlines.open(path) as reader:
                for doc in reader:
                    docs.append(doc)
        return docs

    def _build_index(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.get('text', '').split() for doc in self.corpus]
            if tokenized:
                self.index = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not installed, run: pip install rank_bm25")

    def retrieve(self, query: str, k: int) -> List[Passage]:
        if self.index is None:
            return []
        scores = self.index.get_scores(query.split())
        top_k = np.argsort(scores)[-k:][::-1]
        return [
            Passage(
                doc_id=self.corpus[i].get('id', str(i)),
                text=self.corpus[i].get('text', ''),
                score=float(scores[i]),
                source=self.corpus[i].get('source', '')
            )
            for i in top_k
        ]


class LLMGenerator:
    def __init__(self, model_name: str, api: str = "openai",
                 temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.api = api
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        if self.api == "openai":
            return self._generate_openai(prompt)
        elif self.api == "anthropic":
            return self._generate_anthropic(prompt)
        else:
            return f"[{self.api} generator not configured]"

    def _generate_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _generate_anthropic(self, prompt: str) -> str:
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class RAGPipeline:
    def __init__(self, retriever, generator, k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.k = k

    def query(self, query: str) -> RAGResult:
        passages = self.retriever.retrieve(query, self.k)
        context = "\n\n".join([f"[{i+1}] {p.text}" for i, p in enumerate(passages)])
        prompt = f"Based on the following passages, answer the question.\n\n{context}\n\nQuestion: {query}\nAnswer:"
        output = self.generator.generate(prompt)
        return RAGResult(query=query, passages=passages, output=output)
