"""
This module defines parameters and configurations for the training process of a deep learning model.
- TrainerParams: Configurations specific to the training process like GPUs, epochs, save paths, etc.
- ModuleParams: Parameters related to the underlying model, including learning rate, decay, base model, etc.
- DataParams: Parameters concerning the data handling like batch size, cross-validation, seed, etc.
- Params: A comprehensive class encapsulating the above parameters.
Note that some parameters can be overridden by values in a YAML file.
"""

import dataclasses
from typing import Optional

from libs.params import ParamsMixIn

pretrained_model_name_default = "distilbert-base-uncased"
sa_pretrained_model_name_default = "distilbert-base-uncased"
wordembedding_path_default = "./wordembedding.npy"


@dataclasses.dataclass(frozen=True)
class TrainerParams(ParamsMixIn):
    num_tpu_cores: Optional[int] = None
    gpus: Optional[int] = None
    epochs: int = 100
    resume_from_checkpoint: Optional[str] = None
    save_dir: str = "../experiments"
    distributed_backend: Optional[str] = None
    num_nodes: int = 1
    accumulate_grad_batches: int = 1
    weights_save_path: Optional[str] = None
    precision: int = 32
    val_check_interval: float = 1.0

    @property
    def checkpoint_callback(self) -> bool:
        return self.weights_save_path is not None


@dataclasses.dataclass(frozen=True)
class ModuleParams(ParamsMixIn):
    """Define ModuleParams here, then the parameters in yaml can be loaded"""

    lr: float = 1e-5
    weight_decay: float = 1e-4
    base_model: str = "MLFBERT"

    ema_decay: Optional[float] = None
    ema_eval_freq: int = 1

    pretrained_model_name: str = pretrained_model_name_default
    sa_pretrained_model_name: str = sa_pretrained_model_name_default

    wordembedding_path: Optional[str] = wordembedding_path_default

    @property
    def use_ema(self) -> bool:
        return self.ema_decay is not None


@dataclasses.dataclass(frozen=True)
class DataParams(ParamsMixIn):
    batch_size: int = 32

    fold: int = 0  # -1 for cross validation
    n_splits: int = 5  # -1 for train all

    mind_path: str = "../MIND_DATASET"
    pretrained_model_name: str = pretrained_model_name_default

    seed: int = 0

    @property
    def do_cv(self) -> bool:
        return self.fold == -1

    @property
    def train_all(self) -> bool:
        return self.n_splits == -1


@dataclasses.dataclass(frozen=True)
class Params(ParamsMixIn):
    module_params: ModuleParams
    trainer_params: TrainerParams
    data_params: DataParams
    note: str = ""

    @property
    def m(self) -> ModuleParams:
        return self.module_params

    @property
    def t(self) -> TrainerParams:
        return self.trainer_params

    @property
    def d(self) -> DataParams:
        return self.data_params

    @property
    def do_cv(self) -> bool:
        return self.d.do_cv

    def copy_for_cv(self):
        conf_orig = self.dict_config()
        return [
            Params.from_dict(
                {
                    **conf_orig,
                    "data_params": {
                        **conf_orig.module_params,
                        "fold": n,
                    },
                }
            )
            for n in range(self.d.n_splits)
        ]


# %%
if __name__ == "__main__":
    # %%
    p = Params.load("params/001.yaml")
    print(p)
    # %%
    for cp in p.copy_for_cv():
        print(cp.pretty())
