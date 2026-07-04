"""Load and expose application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass
class DataConfig:
    repo_url: str
    huggingface_dataset: str
    raw_dir: Path
    eval_csv: Path
    reports_dir: Path
    indices_dir: Path
    results_dir: Path
    years: list[int]
    document_format: str


@dataclass
class PipelineConfig:
    chunk_size: int
    chunk_overlap: int
    tokenizer: str
    chunk_strategy: str
    embedding_model: str
    vector_db: str
    collection_name: str
    top_k: int
    metadata_filtering: bool = False
    hybrid_search: bool = False
    bm25_weight: float = 0.4
    vector_weight: float = 0.6
    query_expansion: bool = False
    reranking: bool = False
    reranker_model: str = ""
    context_compression: bool = False
    retrieve_candidates: int = 5
    embed_batch_size: int = 128
    chroma_batch_size: int = 512


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int


@dataclass
class EvaluationConfig:
    top_k: int
    factual_tolerance: float
    max_eval_samples: int | None


@dataclass
class PromptConfig:
    baseline: str
    engineered: str


@dataclass
class Settings:
    data: DataConfig
    baseline: PipelineConfig
    engineered: PipelineConfig
    llm: LLMConfig
    evaluation: EvaluationConfig
    prompts: PromptConfig
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    def ensure_directories(self) -> None:
        for path in (
            self.data.raw_dir,
            self.data.reports_dir,
            self.data.indices_dir,
            self.data.results_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _build_pipeline_config(raw: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(**raw)


def load_settings(config_path: Path | None = None) -> Settings:
    path = config_path or CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    data_raw = raw["data"]
    data = DataConfig(
        repo_url=data_raw["repo_url"],
        huggingface_dataset=data_raw["huggingface_dataset"],
        raw_dir=_resolve_path(data_raw["raw_dir"]),
        eval_csv=_resolve_path(data_raw["eval_csv"]),
        reports_dir=_resolve_path(data_raw["reports_dir"]),
        indices_dir=_resolve_path(data_raw["indices_dir"]),
        results_dir=_resolve_path(data_raw["results_dir"]),
        years=[int(y) for y in data_raw["years"]],
        document_format=data_raw["document_format"],
    )

    llm_raw = raw["llm"]
    llm = LLMConfig(**llm_raw)

    eval_raw = raw["evaluation"]
    evaluation = EvaluationConfig(
        top_k=int(eval_raw["top_k"]),
        factual_tolerance=float(eval_raw["factual_tolerance"]),
        max_eval_samples=eval_raw.get("max_eval_samples"),
    )

    prompts = PromptConfig(**raw["prompts"])

    return Settings(
        data=data,
        baseline=_build_pipeline_config(raw["baseline"]),
        engineered=_build_pipeline_config(raw["engineered"]),
        llm=llm,
        evaluation=evaluation,
        prompts=prompts,
    )
