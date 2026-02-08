from musubi_eval.application.run_scenario import ScenarioRunner
from musubi_eval.config import ScenarioConfig
from musubi_eval.infrastructure.dataset_jsonl import JsonlDatasetReader
from musubi_eval.infrastructure.musubi_http import MusubiHttpClient
from musubi_eval.infrastructure.results_filesystem import save_results


def run_scenario(cfg: ScenarioConfig):
    runner = ScenarioRunner(
        dataset_reader=JsonlDatasetReader(),
        search_gateway=MusubiHttpClient(cfg.base_url, cfg.timeout_sec, cfg.retry),
    )
    return runner.run(cfg)


__all__ = ["run_scenario", "save_results"]
