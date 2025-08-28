"""

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_multi_job

Currently runs two of the same job in parallel, each with half the available GPUs.

"""

from ray.util.placement_group import placement_group, PlacementGroup

from transformers import AutoTokenizer
from skyrl_train.dataset import PromptDataset
from skyrl_train.utils import validate_cfg

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from skyrl_train.utils.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.base import GeneratorInterface
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import ray

import os
import hydra
from loguru import logger
from skyrl_train.utils.tracking import Tracking
import multiprocessing as mp

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from pprint import pprint


# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]

def pretty_print_pg(pg: PlacementGroup):
    """Pretty print a Ray PlacementGroup object with actual requested resources."""
    if not isinstance(pg, PlacementGroup):
        raise TypeError("Expected a ray.util.placement_group.PlacementGroup object")

    info = ray.util.placement_group_table(pg)

    print(f"Placement Group '{info.get('name', '')}' (id={pg.id})")
    print(f"  State:    {info.get('state', 'UNKNOWN')}")
    print(f"  Strategy: {info.get('strategy', 'UNKNOWN')}")
    print(f"  Bundles:")

    for i, bundle in enumerate(info.get("bundles", [])):
        # bundle is a dict of resources -> amount
        if isinstance(bundle, dict) and bundle:
            print(f"    Bundle {i}: {bundle}")
        else:
            # fallback if bundle info is missing
            print(f"    Bundle {i}: (requested: {pg.bundle_specs[i]})")

def get_multi_job_placement_groups(cfg: DictConfig, num_jobs: int = 2, timeout: int = 180):
    """
    Naively splits available nodes/GPUs in half for each job.
    If enough nodes are available, splits by node; otherwise, splits GPUs on the same node.
    Returns a list of PlacementGroups, one per job.
    """

    # Determine total required GPUs for all jobs
    total_gpus = (
        cfg.generator.num_inference_engines
        * cfg.generator.inference_engine_tensor_parallel_size
    )
    # Get available resources from Ray
    resources = ray.cluster_resources()
    available_gpus = int(resources.get("GPU", 0))
    available_nodes = [
        node["NodeID"] for node in ray.nodes() if node.get("Alive", False)
    ] if hasattr(ray, "nodes") else []

    # Naive split: try to split by node if enough nodes, else split GPUs
    pgs = []
    if available_nodes and len(available_nodes) >= num_jobs:
        # Split by node: each job gets all GPUs on its node
        gpus_per_job = available_gpus // num_jobs
        for _ in range(num_jobs):
            pg = placement_group(
                [{"GPU": gpus_per_job, "CPU": gpus_per_job}],
                strategy="STRICT_SPREAD",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            pgs.append(pg)
    else:
        # Split GPUs on the same node
        gpus_per_job = total_gpus // num_jobs
        for _ in range(num_jobs):
            pg = placement_group(
                [{"GPU": gpus_per_job, "CPU": gpus_per_job}],
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            pgs.append(pg)

    return pgs

@ray.remote(num_cpus=1)
def run_job(cfg: DictConfig, pg):
    exp = BasePPOExp(cfg)
    exp.colocate_pg = pg
    exp.run()

@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    pprint(cfg)

    initialize_ray(cfg)
    pgs = get_multi_job_placement_groups(cfg, num_jobs=2)

    pretty_print_pg(pgs[0])
    pretty_print_pg(pgs[1])

    cfg.trainer.placement.policy_num_gpus_per_node //= 2
    cfg.trainer.placement.ref_num_gpus_per_node //= 2
    cfg.generator.num_inference_engines //= 2

    # ray.get(run_job.remote(cfg, pgs[0]))

    ray.get([
        run_job.remote(cfg, pgs[0]),
        run_job.remote(cfg, pgs[1]),
    ])

if __name__ == "__main__":
    main()

