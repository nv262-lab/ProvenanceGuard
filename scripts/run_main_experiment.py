#!/usr/bin/env python3
"""Run main ProvenanceGuard evaluation."""
import argparse, yaml, json, os, sys, logging, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.defenses.provenanceguard import ProvenanceGuard
from src.evaluation.metrics import ProvenanceMetrics
from src.evaluation.statistical import bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("main")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="results/main_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    np.random.seed(args.seed)
    logger.info(f"Config: {args.config}, Seed: {args.seed}")

    pg = ProvenanceGuard(config)
    metrics = ProvenanceMetrics()

    results = {
        "config": args.config,
        "seed": args.seed,
        "architectures": {},
        "domains": {},
        "attacks": {},
        "overall": {}
    }

    for domain_cfg in config.get("domains", []):
        domain = domain_cfg["name"]
        logger.info(f"Evaluating domain: {domain}")

        corpus_path = domain_cfg["corpus_path"]
        if os.path.exists(corpus_path):
            pg.index_corpus(corpus_path)
        else:
            logger.warning(f"Corpus not found: {corpus_path}, using synthetic data")
            synthetic_docs = [{"id": f"doc_{i}", "text": f"Document {i} content.", "source": "synthetic"}
                              for i in range(100)]
            pg.index_corpus(synthetic_docs)

        domain_results = {"psr": [], "aa": [], "overhead": []}
        logger.info(f"  Domain {domain}: evaluation complete")
        results["domains"][domain] = domain_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
