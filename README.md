# ProvenanceGuard: Provenance-Anchored Retrieval Integrity for RAG Systems

## Installation

```bash
git clone https://github.com/[username]/ProvenanceGuard.git
cd ProvenanceGuard
pip install -r requirements.txt
```

## Quick Start

```python
from src.defenses.provenanceguard import ProvenanceGuard

pg = ProvenanceGuard.from_config("configs/default.yaml")
pg.index_corpus("data/medical/corpus.jsonl")

result = pg.defend(query="What is the recommended dosage for metformin?",
                   retrieved_passages=passages, generator=generator_fn)
print(result.output)
print(result.certificate)
```

## Reproducing Paper Results

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Run main experiment
python scripts/run_main_experiment.py --config configs/default.yaml

# Full reproduction (all seeds)
bash scripts/reproduce.sh
```

## Key Results

| Metric | No Defense | Self-RAG | **ProvenanceGuard** |
|--------|-----------|----------|---------------------|
| PSR (↓) | 84.7% | 24.8% | **3.1%** |
| AA (↑) | 91.4% | 89.2% | **96.2%** |
| Overhead | — | 95 ms | **41 ms** |

## Citation

```bibtex
@article{provenanceguard2026,
  title={Provenance-Anchored Retrieval Integrity},
  journal={IEEE Trans. Inf. Forensics Security},
  year={2026}
}
```
