
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 100
    n_val: int = 32
    filter_train: bool = True
    filter_val: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    max_new_tokens: int = 512
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "llamaguard2")
    refusal_eval_methodologies: Tuple[str] = ("substring_matching",)
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048

    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)