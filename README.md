# ProvenanceGuard: Provenance-Anchored Retrieval Integrity for RAG Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Authors:** Naga Sujitha Vummaneni¹*, Usha Ratnam Jammula²  
¹ Cornell University, nv262@cornell.edu (*corresponding author)  
² Independent Researcher, jammula.usha@gmail.com

**Paper:** IEEE Transactions on Information Forensics and Security (TIFS), 2026

## Overview

ProvenanceGuard is an information-theoretic framework that establishes verifiable guarantees on the semantic trustworthiness of Retrieval-Augmented Generation (RAG) outputs. It implements three defense mechanisms:

- **CPB (Cryptographic Provenance Binding)**: Anchors retrieved content to tamper-evident semantic fingerprints
- **MIG (Mutual Information Gating)**: Eliminates passages whose information-theoretic contribution deviates from trusted baselines
- **ARD (Adaptive Retrieval Diversification)**: Randomizes the retrieval surface to prevent targeted poisoning

## Key Results

| Metric | No Defense | Self-RAG | **ProvenanceGuard** |
|--------|-----------|----------|---------------------|
| PSR (↓) | 84.7% | 24.8% | **3.1%** |
| AA (↑) | 91.4% | 89.2% | **96.2%** |
| Overhead | — | 95 ms | **41 ms** |

## Installation

```bash
git clone https://github.com/nv262-lab/ProvenanceGuard.git
cd ProvenanceGuard
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy<2" "torch" "transformers==4.40.0" "sentence-transformers==2.7.0"
pip install -r requirements.txt
```

## Quick Start

```python
from src.defenses.provenanceguard import ProvenanceGuard
from src.utils.rag_pipeline import Passage
import jsonlines

pg = ProvenanceGuard.from_config("configs/default.yaml")

with jsonlines.open("data/medical/corpus.jsonl") as r:
    docs = list(r)
pg.index_corpus(docs)

passages = [
    Passage(doc_id="medical_00001", text="Metformin dosage: 500mg daily.", score=0.9),
    Passage(doc_id="FAKE_001", text="FABRICATED: Take 5000mg metformin.", score=0.95),
]

result = pg.defend(query="What is the recommended metformin dosage?",
                   retrieved_passages=passages)

print("Verified:", len(result.verified_passages))   # 1
print("Flagged:", len(result.flagged_passages))      # 1
print("Certificate:", result.certificate)
print(f"Overhead: {result.overhead_ms:.1f}ms")       # ~37ms
```

## Reproducing Paper Results

```bash
python3 scripts/generate_sample_data.py

python3 scripts/run_main_experiment.py --seed 42 --output results/seed_42.json
python3 scripts/run_main_experiment.py --seed 123 --output results/seed_123.json
python3 scripts/run_main_experiment.py --seed 456 --output results/seed_456.json

python3 -m pytest tests/ -v
```

## Project Structure

```
ProvenanceGuard/
├── src/
│   ├── defenses/          # CPB, MIG, ARD, combined pipeline
│   ├── attacks/           # TFP, SCP, RM, MCP attack strategies
│   ├── evaluation/        # Metrics (PSR, AA, PS, FRR) + statistics
│   └── utils/             # Embeddings, fingerprinting, RAG pipeline
├── configs/default.yaml   # All hyperparameters
├── scripts/               # Reproduction scripts
├── data/                  # Sample datasets
├── tests/                 # Unit tests
└── dataport/              # IEEE DataPort documentation
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_H` | 0.15 | CPB Hamming distance threshold |
| `tau_c` | 0.6 | MIG MI ratio threshold |
| `K` | 15 | ARD extended pool size |
| `beta` | 2.0 | ARD noise scale |
| `k` | 5 | Number of retrieved passages |

## Citation

```bibtex
@article{vummaneni2026provenanceguard,
  title={Provenance-Anchored Retrieval Integrity: An Information-Theoretic
         Framework for Defending RAG Systems Against Retrieval Poisoning},
  author={Vummaneni, Naga Sujitha and Jammula, Usha Ratnam},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
