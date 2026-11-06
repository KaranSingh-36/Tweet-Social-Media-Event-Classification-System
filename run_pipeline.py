"""Run the complete NLP pipeline end to end.

    python run_pipeline.py                  # everything (DistilBERT takes ~45-60 min on CPU)
    python run_pipeline.py --skip-transformer
    python run_pipeline.py --only evaluate

Steps: data -> explore -> classical -> transformer -> evaluate
"""
from __future__ import annotations

import argparse
import time

STEPS = ["data", "explore", "classical", "transformer", "evaluate"]


def run(step: str) -> None:
    start = time.perf_counter()
    print(f"\n{'=' * 70}\n>>> {step.upper()}\n{'=' * 70}")
    if step == "data":
        from src.data_loader import prepare_datasets

        prepare_datasets()
    elif step == "explore":
        from src.data_exploration import main

        main()
    elif step == "classical":
        from src.classical_model import train_classical_models

        train_classical_models()
    elif step == "transformer":
        from src.transformer_model import train_transformer

        train_transformer()
    elif step == "evaluate":
        from src.evaluation import evaluate_all

        evaluate_all()
    print(f"<<< {step} finished in {time.perf_counter() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=STEPS, nargs="+", help="run only these steps")
    parser.add_argument("--skip-transformer", action="store_true", help="skip DistilBERT fine-tuning")
    args = parser.parse_args()
    steps = args.only or [s for s in STEPS if not (args.skip_transformer and s == "transformer")]
    for step in steps:
        run(step)


if __name__ == "__main__":
    main()
