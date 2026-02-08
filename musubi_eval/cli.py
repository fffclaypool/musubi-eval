import argparse
import logging
from pathlib import Path

from .config import load_scenario
from .pipeline import run_scenario, save_results
from .reporting import generate_evidently_report, log_mlflow
from .util import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="musubi evaluation runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run an evaluation scenario")
    run_p.add_argument("-c", "--config", required=True, help="path to scenario YAML")

    args = parser.parse_args()

    if args.cmd == "run":
        cfg = load_scenario(args.config)
        setup_logging(cfg.log_level)
        logger.info("starting scenario: %s", args.config)
        results = run_scenario(cfg)
        outputs = save_results(cfg, results)
        report_outputs = {}
        if cfg.evidently.enabled:
            report_outputs = generate_evidently_report(
                results, cfg.evidently, Path(cfg.evidently.output_dir)
            )
        if cfg.mlflow.enabled:
            log_mlflow(results, cfg.mlflow, cfg, {**outputs, **report_outputs})
        logger.info("results saved: %s", outputs)


if __name__ == "__main__":
    main()
