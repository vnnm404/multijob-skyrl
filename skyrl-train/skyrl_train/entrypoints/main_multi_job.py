"""

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_multi_job

Currently runs two of the same job in parallel, each with half the available GPUs.

"""

import hydra
from omegaconf import DictConfig, OmegaConf
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.utils.utils import initialize_ray
import ray
from ray.util.placement_group import placement_group


# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


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
        node for node, res in ray.nodes().items() if res["Alive"]
    ] if hasattr(ray, "nodes") else None

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
    initialize_ray(cfg)
    pgs = get_multi_job_placement_groups(cfg, num_jobs=2)
    ray.get([
        run_job.remote(cfg, pgs[0]),
        run_job.remote(cfg, pgs[1]),
    ])

if __name__ == "__main__":
    main()
