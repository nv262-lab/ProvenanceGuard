"""ProvenanceGuard: Combined CPB + MIG + ARD defense pipeline."""
import yaml
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .cpb import CPBDefense
from .mig import MIGDefense
from .ard import ARDDefense
from ..utils.embeddings import EmbeddingModel
from ..utils.rag_pipeline import RAGResult, Passage

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceCertificate:
    """Provenance certificate for a RAG output."""
    query: str
    verified_passages: List[Dict] = field(default_factory=list)
    flagged_passages: List[Dict] = field(default_factory=list)
    cpb_results: List[Dict] = field(default_factory=list)
    mig_results: List[Dict] = field(default_factory=list)
    ard_applied: bool = False
    total_overhead_ms: float = 0.0
    integrity_score: float = 0.0

    def to_dict(self):
        return {
            "query": self.query,
            "n_verified": len(self.verified_passages),
            "n_flagged": len(self.flagged_passages),
            "integrity_score": self.integrity_score,
            "overhead_ms": self.total_overhead_ms,
            "cpb_pass_rate": sum(1 for r in self.cpb_results if r["verified"]) / max(len(self.cpb_results), 1),
            "mig_consistency_rate": sum(1 for r in self.mig_results if r["consistent"]) / max(len(self.mig_results), 1),
        }


class ProvenanceGuard:
    """Combined defense pipeline: CPB -> MIG -> ARD -> Certificate."""

    def __init__(self, config: dict):
        self.config = config
        self.embedder = EmbeddingModel(
            model_name=config.get("cpb", {}).get("embedding_model",
                        "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.cpb = CPBDefense(
            self.embedder,
            num_bits=config.get("cpb", {}).get("simhash_bits", 256),
            hamming_threshold=config.get("cpb", {}).get("hamming_threshold", 0.15)
        )
        self.mig = MIGDefense(
            self.embedder,
            mi_threshold=config.get("mig", {}).get("mi_threshold", 0.6)
        )
        self.ard = ARDDefense(
            pool_multiplier=config.get("ard", {}).get("pool_multiplier", 3),
            noise_scale=config.get("ard", {}).get("noise_scale", 2.0)
        )

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def index_corpus(self, corpus_path_or_docs):
        """Index corpus documents for CPB verification."""
        if isinstance(corpus_path_or_docs, str):
            import jsonlines
            with jsonlines.open(corpus_path_or_docs) as reader:
                docs = list(reader)
        else:
            docs = corpus_path_or_docs
        self.cpb.index_corpus(docs)

    def defend(self, query: str, retrieved_passages: List[Passage],
               generator=None) -> RAGResult:
        """Apply full defense pipeline to a RAG query."""
        start = time.time()
        cert = ProvenanceCertificate(query=query)

        # Stage 1: CPB verification
        cpb_results = self.cpb.verify_passages(retrieved_passages)
        cert.cpb_results = cpb_results
        verified = [r["passage"] for r in cpb_results if r["verified"]]
        flagged = [r["passage"] for r in cpb_results if not r["verified"]]

        # Stage 2: MIG consistency check on verified passages
        if verified:
            mig_results = self.mig.filter_passages(verified, query)
            cert.mig_results = mig_results
            consistent = [r["passage"] for r in mig_results if r["consistent"]]
            flagged.extend([r["passage"] for r in mig_results if not r["consistent"]])
        else:
            consistent = []

        # Stage 3: ARD diversification
        if consistent and len(consistent) > 1:
            embeddings = self.embedder.encode([p.text for p in consistent])
            query_emb = self.embedder.encode(query)[0]
            k = min(self.config.get("rag", {}).get("retrieval_k", 5), len(consistent))
            final_passages = self.ard.diversify(consistent, embeddings, query_emb, k)
            cert.ard_applied = True
        else:
            final_passages = consistent

        # Generate output
        output = ""
        if generator and final_passages:
            context = "\n\n".join([f"[{i+1}] {p.text}" for i, p in enumerate(final_passages)])
            prompt = f"Based on the following verified passages, answer:\n\n{context}\n\nQuestion: {query}\nAnswer:"
            output = generator.generate(prompt)

        # Compute certificate
        cert.verified_passages = [{"doc_id": p.doc_id, "text": p.text[:100]} for p in final_passages]
        cert.flagged_passages = [{"doc_id": p.doc_id, "reason": "cpb/mig"} for p in flagged]
        cert.integrity_score = len(final_passages) / max(len(retrieved_passages), 1)
        cert.total_overhead_ms = (time.time() - start) * 1000

        return RAGResult(
            query=query, passages=final_passages, output=output,
            certificate=cert.to_dict(), flagged_passages=flagged,
            verified_passages=final_passages, overhead_ms=cert.total_overhead_ms
        )
