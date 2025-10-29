import csv
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence, cast

import cupy as cp
import numpy as np
import cubewalkers as cw
from cana.boolean_network import BooleanNetwork
from scipy import integrate
from sklearn.metrics import matthews_corrcoef
import random


DEFAULT_NOISES = (
    0.000,
    0.010,
    0.020,
    0.030,
    0.040,
    0.050,
    0.060,
    0.070,
    0.080,
    0.090,
    0.100,
)


def ca_network_from_output_list(output_list, k=7, lattice=149):
    """
    For a given Boolean LUT, creates a logic dictionary for a CA network. Then, instantiates a BooleanNetwork object from the logic dictionary.
    Logic dict format:
        logic = {
            1: {'name': '1', 'in': [0, 1, 2], 'out': ['0', '1', '0', '1', '1', '0', '1', '0']},
            2: {'name': '2', 'in': [1, 2, 0], 'out': ['0', '1', '0', '1', '1', '0', '1', '0']},
            0: {'name': '0', 'in': [2, 0, 1], 'out': ['0', '1', '0', '1', '1', '0', '1', '0']}
        }

    """
    indices = []
    for i in range(lattice):
        index = []
        for j in range(k):
            index.append((i + j) % lattice)
        indices.append(index)
    logic = {}
    for i in range(lattice):
        middle_cell = indices[i][len(indices[i]) // 2]
        logic[middle_cell] = {
            "name": str(middle_cell),
            "in": indices[i],
            "out": output_list,
        }
    return BooleanNetwork.from_dict(logic)


def compute_accuracy(model, noise=0.0, deviations=3.0):
    n_steps, n_cells, n_samples = model.trajectories.shape

    epsilon = (
        noise * n_cells + (noise * (1 - noise) * n_cells) ** (1 / 2) * deviations
    ) // 1

    midpoint = n_cells // 2 + 1
    start_above = model.initial_states[:, :].sum(axis=0) >= midpoint
    start_below = model.initial_states[:, :].sum(axis=0) < midpoint

    # old convergence : takes only one timestep to determine convergence
    # end_above = model.trajectories[-1, :, :].sum(axis=0) >= (n_cells - epsilon)
    # end_below = model.trajectories[-1, :, :].sum(axis=0) <= epsilon

    # new convergence measure that takes 2 consecutive time steps to determine convergence
    end_above = (model.trajectories[-1, :, :].sum(axis=0) >= (n_cells - epsilon)) & (
        model.trajectories[-2, :, :].sum(axis=0) >= (n_cells - epsilon)
    )
    end_below = (model.trajectories[-1, :, :].sum(axis=0) <= epsilon) & (
        model.trajectories[-2, :, :].sum(axis=0) <= epsilon
    )

    true_positive = (start_above & end_above).sum()
    false_positive = (start_below & end_above).sum()
    true_negative = (start_below & end_below).sum()
    false_negative = (start_above & end_below).sum()
    non_converge_start_above = (start_above & ~(end_above | end_below)).mean()
    non_converge_start_below = (start_below & ~(end_above | end_below)).mean()

    precision0 = (
        true_negative / (true_negative + false_negative)
        if (true_negative + false_negative) != 0
        else 0
    )  # negative predictive value NPV
    precision1 = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) != 0
        else 0
    )
    recall0 = (
        true_negative / (start_below).sum() if (start_below).sum() != 0 else 0
    )  # specificity
    recall1 = (
        true_positive / (start_above).sum() if (start_above).sum() != 0 else 0
    )  # sensitivity
    f1 = (
        2 * (precision1 * recall1) / (precision1 + recall1)
        if (precision1 + recall1) != 0
        else 0
    )

    p1 = true_positive / (start_above).sum()
    p0 = true_negative / (start_below).sum()
    p05 = ((start_above & end_above) | (start_below & end_below)).sum() / n_samples

    # scikit's mcc
    # Define three classes: 1 for above, -1 for below, 0 for non-convergence
    y_true = np.zeros(n_samples, dtype=int)
    y_true[start_above.get()] = 1
    y_true[start_below.get()] = -1
    y_pred = np.zeros(n_samples, dtype=int)
    y_pred[end_above.get()] = 1
    y_pred[end_below.get()] = -1
    # Calculate MCC for multi-class
    mcc = matthews_corrcoef(y_true, y_pred)

    accuracy = {
        "majority0": float(p0),
        "majority1": float(p1),
        "P0.5": float(p05),
        "precision0": float(precision0),
        "precision1": float(precision1),
        "recall0": float(recall0),
        "recall1": float(recall1),
        "nonconverge1": float(non_converge_start_above),
        "nonconverge0": float(non_converge_start_below),
        "f1": float(f1),
        "mcc": float(mcc),
        "minP0P1": float(min(p0, p1)),
    }

    return accuracy


# Internal container for staged simulations when streaming batches.
@dataclass
class _MultiRuleSimulationJob:
    stream: cp.cuda.Stream
    event: cp.cuda.Event
    trajectories: cp.ndarray
    initial_states: cp.ndarray
    rules: tuple[str, ...]
    noises: tuple[float, ...]
    n_walkers: int
    lattice_size: int


# Launch a multi-rule simulation on a dedicated CUDA stream so we can overlap batches.
def _launch_multi_rule_multi_noise_job(
    *,
    rules: Sequence[str],
    noises: Sequence[float],
    n_walkers: int,
    time_step_factor: int,
    lattice_size: int,
    k: int,
    initial_config_type: str,
    device_id: int,
    threads_per_block: tuple[int, int],
    T_window: int,
    stream: cp.cuda.Stream | None = None,
) -> _MultiRuleSimulationJob:
    rules_tuple = tuple(rules)
    noises_tuple = tuple(float(n) for n in noises)

    if not rules_tuple:
        raise ValueError("rules must contain at least one rule")
    if not noises_tuple:
        raise ValueError("noises must contain at least one value")

    stream_obj = stream

    with cp.cuda.Device(device_id):
        if stream_obj is None:
            stream_obj = cp.cuda.Stream(non_blocking=True)
        with stream_obj:
            lookup_tables_list = []
            node_regulators = None

            for rule in rules_tuple:
                output_list = list(rule)
                network = ca_network_from_output_list(
                    output_list, k=k, lattice=lattice_size
                )

                rule_luts = []
                for noise in noises_tuple:
                    outs, ins = cw.conversions.cana2cupy_probabilisticLUT(
                        network, prob=noise
                    )
                    rule_luts.append(outs)
                    if node_regulators is None:
                        node_regulators = ins
                    else:
                        if not cp.all(ins == node_regulators):
                            raise ValueError(
                                "Node regulators differ across noise variants or rules"
                            )
                lookup_tables_list.append(cp.stack(rule_luts, axis=0))

            assert node_regulators is not None
            lookup_tables_rules_noises = cp.stack(lookup_tables_list, axis=0)

            if initial_config_type not in ["normal", "uniform_bias_dist"]:
                raise ValueError(
                    "initial_config_type must be 'normal' or 'uniform_bias_dist'"
                )

            num_rules = len(rules_tuple)
            num_noises = len(noises_tuple)
            total_walkers = n_walkers * num_rules * num_noises

            walker_rule_idx = cp.repeat(
                cp.arange(num_rules, dtype=cp.int32),
                num_noises * n_walkers,
            )
            walker_noise_idx = cp.tile(
                cp.repeat(cp.arange(num_noises, dtype=cp.int32), n_walkers),
                num_rules,
            )

            if initial_config_type == "uniform_bias_dist":
                biases = cp.linspace(0, 1, total_walkers)
                init_states = (
                    cp.random.random((lattice_size, total_walkers)) <= biases
                ).astype(cp.bool_)
            else:
                init_states = (
                    cp.random.random((lattice_size, total_walkers)) <= 0.5
                ).astype(cp.bool_)

            n_time_steps = lattice_size * time_step_factor + 1

            trajectories, used_initial_states, _, _ = (
                simulate_multi_rule_multi_noise_probabilistic(
                    lookup_tables_rules_noises=lookup_tables_rules_noises,
                    node_regulators=node_regulators,
                    N=lattice_size,
                    T=n_time_steps,
                    W=total_walkers,
                    walker_rule_idx=walker_rule_idx,
                    walker_noise_idx=walker_noise_idx,
                    maskfunction=cw.update_schemes.synchronous_PBN,
                    T_window=T_window,
                    threads_per_block=threads_per_block,
                    initial_states=init_states,
                    device_id=device_id,
                )
            )

        event = cp.cuda.Event()
        event.record(stream_obj)

    assert stream_obj is not None

    return _MultiRuleSimulationJob(
        stream=stream_obj,
        event=event,
        trajectories=trajectories,
        initial_states=used_initial_states,
        rules=rules_tuple,
        noises=noises_tuple,
        n_walkers=n_walkers,
        lattice_size=lattice_size,
    )


def _finalize_multi_rule_multi_noise_job(
    job: _MultiRuleSimulationJob,
) -> dict[str, dict[str, dict[float | str, float]]]:
    job.event.synchronize()

    trajectories = job.trajectories
    initial_states = job.initial_states
    window_len, node_count, total_walkers = trajectories.shape
    num_rules = len(job.rules)
    num_noises = len(job.noises)

    if total_walkers != job.n_walkers * num_rules * num_noises:
        raise ValueError("Walker layout is incompatible with requested reshape")

    try:
        traj_view = trajectories.reshape(
            window_len,
            node_count,
            num_rules,
            num_noises,
            job.n_walkers,
        )
        init_view = initial_states.reshape(
            node_count,
            num_rules,
            num_noises,
            job.n_walkers,
        )
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            "Walker layout is incompatible with reshape; check n_walkers"
        ) from exc

    results: dict[str, dict[float, dict[str, float]]] = {rule: {} for rule in job.rules}

    for rule_idx, rule in enumerate(job.rules):
        for noise_idx, noise in enumerate(job.noises):
            traj_slice = traj_view[:, :, rule_idx, noise_idx, :]
            init_slice = init_view[:, rule_idx, noise_idx, :]

            model_stub = SimpleNamespace(
                trajectories=traj_slice,
                initial_states=init_slice,
            )

            results[rule][noise] = compute_accuracy(
                model_stub,
                noise=noise,
                deviations=3.0,
            )

    metric_names = [
        "majority0",
        "majority1",
        "P0.5",
        "precision0",
        "precision1",
        "recall0",
        "recall1",
        "nonconverge1",
        "nonconverge0",
        "f1",
        "mcc",
        "minP0P1",
    ]

    final_results: dict[str, dict[str, dict[float | str, float]]] = {}

    x = np.array(job.noises, dtype=float)

    for rule in job.rules:
        if rule not in results or not results[rule]:
            final_results[rule] = {"AUC": {}}
            continue

        accuracy_dict: dict[str, dict[float | str, float]] = {
            metric: {} for metric in metric_names
        }
        accuracy_dict["AUC"] = {}

        for noise in job.noises:
            if noise not in results[rule]:
                continue
            metric_values = results[rule][noise]
            for metric_name, score in metric_values.items():
                accuracy_dict[metric_name][noise] = float(score)

        for metric_name in metric_names:
            y = np.array(
                [accuracy_dict[metric_name].get(noise, 0.0) for noise in job.noises],
                dtype=float,
            )
            auc = float(integrate.trapezoid(y, x=x))
            accuracy_dict[metric_name]["AUC"] = auc
            accuracy_dict["AUC"][metric_name] = auc

        try:
            accuracy_dict["AUC"]["minAUC"] = min(
                accuracy_dict["AUC"][metric]
                for metric in ("majority0", "majority1", "P0.5")
                if metric in accuracy_dict["AUC"]
            )
        except ValueError:
            pass

        final_results[rule] = accuracy_dict

    gc.collect()
    return final_results


# Multi-variant probabilistic LUT simulation (new functionality)
# new fitness func that computes all noise levels in one go in the cupy kernel

_MULTI_VARIANT_KERNEL: cp.RawKernel | None = None


def _get_multi_variant_kernel() -> cp.RawKernel:
    """Compile or retrieve the cached multi-variant probabilistic LUT kernel."""

    global _MULTI_VARIANT_KERNEL
    if _MULTI_VARIANT_KERNEL is None:
        kernel_body = r"""
extern "C" __global__
void multi_variant_probabilistic(
    const bool* input_state,
    const float* mask,
    bool* output_state,
    const float* lut,
    const int* regulators,
    const int* walker_variant,
    int N,
    int W,
    int L,
    int max_inputs)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int n = blockDim.y * blockIdx.y + threadIdx.y;
    if (w >= W || n >= N) {
        return;
    }

    int idx = n * W + w;
    if (mask[idx] <= 0.0f) {
        output_state[idx] = input_state[idx];
        return;
    }

    int variant = walker_variant[w];
    long long lut_offset = ((long long)variant * N + n) * L;
    int lookup_index = 0;
    int reg_offset = n * max_inputs;

    for (int k = 0; k < max_inputs; ++k) {
        int regulator = regulators[reg_offset + k];
        if (regulator < 0) {
            break;
        }
        lookup_index = (lookup_index << 1) + (int)input_state[regulator * W + w];
    }

    float threshold = lut[lut_offset + lookup_index];
    output_state[idx] = threshold >= mask[idx];
}
"""

        _MULTI_VARIANT_KERNEL = cp.RawKernel(
            kernel_body,
            "multi_variant_probabilistic",
        )  # type: ignore[arg-type]
    return _MULTI_VARIANT_KERNEL


def simulate_multi_variant_probabilistic(
    *,
    lookup_tables_variants: cp.ndarray,
    node_regulators: cp.ndarray,
    N: int,
    T: int,
    W: int,
    walker_variant_idx: cp.ndarray | None = None,
    maskfunction=cw.update_schemes.synchronous_PBN,
    T_window: int | None = None,
    threads_per_block: tuple[int, int] = (16, 16),
    initial_states: cp.ndarray | None = None,
    device_id: int | None = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Simulate an ensemble where walkers are partitioned across LUT variants.

    Parameters
    ----------
    lookup_tables_variants : cp.ndarray
        Array of shape (V, N, L) containing probabilistic LUTs for each variant.
    node_regulators : cp.ndarray
        Integer array of shape (N, max_inputs) describing regulators per node.
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Total number of walkers.
    walker_variant_idx : cp.ndarray | None, optional
        Mapping from walker index to LUT variant. If ``None``, walkers are
        distributed as evenly as possible across variants.
    maskfunction : callable, optional
        Update scheme that produces the update mask. Defaults to synchronous PBN.
    T_window : int | None, optional
        Number of trailing timesteps to retain. If ``None`` or invalid, all
        timesteps are retained.
    threads_per_block : tuple[int, int], optional
        CUDA block shape (node_dim, walker_dim). Defaults to (16, 16).
    initial_states : cp.ndarray | None, optional
        Optional initial state array of shape (N, W). If ``None``, random states
        are generated.

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray, cp.ndarray]
        ``(trajectories, initial_states, walker_variant_idx)`` where
        ``trajectories`` has shape (T_window, N, W).
    """
    with cp.cuda.Device(device_id):
        if lookup_tables_variants.ndim != 3:
            raise ValueError("lookup_tables_variants must have shape (variants, N, L)")

        num_variants, lut_nodes, lut_length = lookup_tables_variants.shape
        if lut_nodes != N:
            raise ValueError("lookup_tables_variants second dimension must equal N")

        max_inputs = node_regulators.shape[1]

        if W <= 0:
            raise ValueError("Number of walkers must be positive")

        if walker_variant_idx is None:
            base = W // num_variants
            remainder = W % num_variants
            assignments = []
            for variant in range(num_variants):
                count = base + (1 if variant < remainder else 0)
                assignments.append(cp.full((count,), np.int32(variant), dtype=cp.int32))
            walker_variant_arr = cp.concatenate(assignments)
            # cp.random.shuffle(walker_variant_arr)
        else:
            walker_variant_arr = cp.asarray(walker_variant_idx, dtype=cp.int32)
            if walker_variant_arr.size != W:
                raise ValueError("walker_variant_idx must have length equal to W")

        if initial_states is None:
            init_states = (cp.random.random((N, W)) <= 0.5).astype(cp.bool_)
        else:
            if initial_states.shape != (N, W):
                raise ValueError("initial_states must have shape (N, W)")
            init_states = initial_states.astype(cp.bool_, copy=False)

        kernel = _get_multi_variant_kernel()
        tables = cp.ascontiguousarray(
            lookup_tables_variants.astype(cp.float32, copy=False)
        )
        regulators = cp.ascontiguousarray(node_regulators.astype(cp.int32, copy=False))
        variant_map = cp.ascontiguousarray(walker_variant_arr)

        tpb_nodes, tpb_walkers = threads_per_block
        if tpb_nodes * tpb_walkers > 1024:
            raise ValueError("threads_per_block product must not exceed 1024")
        block = (int(tpb_walkers), int(tpb_nodes))
        grid = (
            (W + tpb_walkers - 1) // tpb_walkers,
            (N + tpb_nodes - 1) // tpb_nodes,
        )

        out = init_states.copy()

        if T_window is None or T_window > T or T_window < 1:
            T_window = T + 1

        trajectories = cp.empty((T_window, N, W), dtype=cp.bool_)
        trajectories[0, :, :] = out

        for t in range(T):
            current = out.copy()
            mask = maskfunction(
                t,
                N,
                W,
                current,
                threads_per_block=threads_per_block,
            ).astype(cp.float32, copy=False)

            kernel(
                grid,
                block,
                (
                    current,
                    mask,
                    out,
                    tables,
                    regulators,
                    variant_map,
                    np.int32(N),
                    np.int32(W),
                    np.int32(lut_length),
                    np.int32(max_inputs),
                ),
            )

            if t >= T - T_window:
                trajectories[t - (T - T_window), :, :] = out

        return trajectories, init_states, variant_map


def fitness_split_multi_variant(
    rule: str,
    noises: Sequence[float] | None = None,
    n_walkers: int = 1000,
    time_step_factor: int = 5,
    lattice_size: int = 149,
    k: int = 7,
    initial_config_type: str = "normal",
    device_id: int = 0,
    threads_per_block: tuple[int, int] = (32, 32),
    T_window: int = 2,
) -> dict[str, dict[float | str, float]]:
    """Evaluate a rule across multiple noise variants in a single simulation.

    Parameters
    ----------
    rule : str
        Rule string to evaluate.
    noises : Sequence[float] | None, optional
        Noise levels to simulate. Defaults to a ten-point sweep if omitted.
    n_walkers : int, optional
        Number of walkers assigned *per noise level*. The total walkers used in
        the simulation are ``n_walkers * len(noises)``.
    device_id : int, optional
        Identifier of the CuPy device on which to execute the simulation.
    """

    if noises is None:
        noises = [
            0.000,  # 0.002, 0.004, 0.006, 0.008,
            0.010,  # 0.012, 0.014, 0.016, 0.018,
            0.020,  # 0.022, 0.024, 0.026, 0.028,
            0.030,  # 0.032, 0.034, 0.036, 0.038,
            0.040,  # 0.042, 0.044, 0.046, 0.048,
            0.050,  # 0.052, 0.054, 0.056, 0.058,
            0.060,  # 0.062, 0.064, 0.066, 0.068,
            0.070,  # 0.072, 0.074, 0.076, 0.078,
            0.080,  # 0.082, 0.084, 0.086, 0.088,
            0.090,  # 0.092, 0.094, 0.096, 0.098,
            0.100,
        ]

    noises = [float(noise) for noise in noises]
    if len(noises) == 0:
        raise ValueError("noises must contain at least one value")

    output_list = list(rule)
    network = ca_network_from_output_list(output_list, k=k, lattice=lattice_size)

    with cp.cuda.Device(device_id):
        lookup_tables = []
        node_regulators = None
        for noise in noises:
            outs, ins = cw.conversions.cana2cupy_probabilisticLUT(network, prob=noise)
            lookup_tables.append(outs)
            if node_regulators is None:
                node_regulators = ins
            else:
                if not cp.all(ins == node_regulators):
                    raise ValueError("Node regulators differ across noise variants")

        assert node_regulators is not None
        lookup_tables_variants = cp.stack(lookup_tables, axis=0)

        if initial_config_type not in ["normal", "uniform_bias_dist"]:
            raise ValueError(
                "initial_config_type must be 'normal' or 'uniform_bias_dist'"
            )

        num_variants = len(noises)
        total_walkers = n_walkers * num_variants

        if initial_config_type == "uniform_bias_dist":
            biases = cp.linspace(0, 1, total_walkers)
            init_states = (
                cp.random.random((lattice_size, total_walkers)) <= biases
            ).astype(cp.bool_)
        else:
            init_states = (
                cp.random.random((lattice_size, total_walkers)) <= 0.5
            ).astype(cp.bool_)

        assignments = [
            cp.full((n_walkers,), np.int32(variant), dtype=cp.int32)
            for variant in range(num_variants)
        ]
        walker_variant_idx = cp.concatenate(assignments)
        # cp.random.shuffle(walker_variant_idx)

        n_time_steps = lattice_size * time_step_factor + 1

        trajectories, used_initial_states, variant_map = (
            simulate_multi_variant_probabilistic(
                lookup_tables_variants=lookup_tables_variants,
                node_regulators=node_regulators,
                N=lattice_size,
                T=n_time_steps,
                W=total_walkers,
                walker_variant_idx=walker_variant_idx,
                maskfunction=cw.update_schemes.synchronous_PBN,
                T_window=T_window,
                threads_per_block=threads_per_block,
                initial_states=init_states,
                device_id=device_id,
            )
        )

        results: dict[float, dict[str, float]] = {}
        for variant, noise in enumerate(noises):
            walker_ids = cp.where(variant_map == variant)[0]
            traj_slice = trajectories[:, :, walker_ids]
            init_slice = used_initial_states[:, walker_ids]
            model_stub = SimpleNamespace(
                trajectories=traj_slice,
                initial_states=init_slice,
            )
            results[noise] = compute_accuracy(
                model_stub,
                noise=noise,
                deviations=3.0,
            )

    if not results:
        return {"AUC": {}}

    metric_names = [
        "majority0",
        "majority1",
        "P0.5",
        "precision0",
        "precision1",
        "recall0",
        "recall1",
        "nonconverge1",
        "nonconverge0",
        "f1",
        "mcc",
        "minP0P1",
    ]
    # metric_names = list(next(iter(results.values())).keys())
    accuracy_dict: dict[str, dict[float | str, float]] = {
        metric: {} for metric in metric_names
    }
    accuracy_dict["AUC"] = {}

    for noise in noises:
        metric_values = results[noise]
        for metric_name, score in metric_values.items():
            accuracy_dict[metric_name][noise] = float(score)

    x = np.array(noises, dtype=float)
    for metric_name in metric_names:
        y = np.array(
            [accuracy_dict[metric_name][noise] for noise in noises], dtype=float
        )
        auc = float(integrate.trapezoid(y, x=x))
        accuracy_dict[metric_name]["AUC"] = auc
        accuracy_dict["AUC"][metric_name] = auc

    try:
        accuracy_dict["AUC"]["minAUC"] = min(
            accuracy_dict["AUC"][metric]
            for metric in ("majority0", "majority1", "P0.5")
            if metric in accuracy_dict["AUC"]
        )
    except ValueError:
        pass

    return accuracy_dict


_MULTI_RULE_MULTI_NOISE_KERNEL: cp.RawKernel | None = None


def _get_multi_rule_multi_noise_kernel() -> cp.RawKernel:
    """Compile or retrieve the cached multi-rule multi-noise probabilistic LUT kernel."""

    global _MULTI_RULE_MULTI_NOISE_KERNEL
    if _MULTI_RULE_MULTI_NOISE_KERNEL is None:
        kernel_body = r"""
extern "C" __global__
void multi_rule_multi_noise_probabilistic(
    const bool* input_state,
    const float* mask,
    bool* output_state,
    const float* lut,
    const int* regulators,
    const int* walker_rule_idx,
    const int* walker_noise_idx,
    int N,
    int W,
    int L,
    int max_inputs,
    int num_noises)
{
    int w = blockDim.x * blockIdx.x + threadIdx.x;
    int n = blockDim.y * blockIdx.y + threadIdx.y;
    if (w >= W || n >= N) {
        return;
    }

    int idx = n * W + w;
    if (mask[idx] <= 0.0f) {
        output_state[idx] = input_state[idx];
        return;
    }

    int rule_idx = walker_rule_idx[w];
    int noise_idx = walker_noise_idx[w];
    // LUT layout: (num_rules, num_noises, N, L)
    long long lut_offset = (((long long)rule_idx * num_noises + noise_idx) * N + n) * L;
    int lookup_index = 0;
    int reg_offset = n * max_inputs;

    for (int k = 0; k < max_inputs; ++k) {
        int regulator = regulators[reg_offset + k];
        if (regulator < 0) {
            break;
        }
        lookup_index = (lookup_index << 1) + (int)input_state[regulator * W + w];
    }

    float threshold = lut[lut_offset + lookup_index];
    output_state[idx] = threshold >= mask[idx];
}
"""

        _MULTI_RULE_MULTI_NOISE_KERNEL = cp.RawKernel(
            kernel_body,
            "multi_rule_multi_noise_probabilistic",
        )  # type: ignore[arg-type]
    return _MULTI_RULE_MULTI_NOISE_KERNEL


def simulate_multi_rule_multi_noise_probabilistic(
    *,
    lookup_tables_rules_noises: cp.ndarray,
    node_regulators: cp.ndarray,
    N: int,
    T: int,
    W: int,
    walker_rule_idx: cp.ndarray,
    walker_noise_idx: cp.ndarray,
    maskfunction=cw.update_schemes.synchronous_PBN,
    T_window: int | None = None,
    threads_per_block: tuple[int, int] = (16, 16),
    initial_states: cp.ndarray | None = None,
    device_id: int | None = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Simulate an ensemble where walkers are partitioned across rule-noise combinations.

    Parameters
    ----------
    lookup_tables_rules_noises : cp.ndarray
        Array of shape (num_rules, num_noises, N, L) containing probabilistic LUTs.
    node_regulators : cp.ndarray
        Integer array of shape (N, max_inputs) describing regulators per node.
    N : int
        Number of nodes in the network.
    T : int
        Number of timesteps to simulate.
    W : int
        Total number of walkers.
    walker_rule_idx : cp.ndarray
        Mapping from walker index to rule variant (shape: W,).
    walker_noise_idx : cp.ndarray
        Mapping from walker index to noise variant (shape: W,).
    maskfunction : callable, optional
        Update scheme that produces the update mask. Defaults to synchronous PBN.
    T_window : int | None, optional
        Number of trailing timesteps to retain. If ``None`` or invalid, all
        timesteps are retained.
    threads_per_block : tuple[int, int], optional
        CUDA block shape (node_dim, walker_dim). Defaults to (16, 16).
    initial_states : cp.ndarray | None, optional
        Optional initial state array of shape (N, W). If ``None``, random states
        are generated.
    device_id : int | None, optional
        GPU device ID to use.

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]
        ``(trajectories, initial_states, walker_rule_idx, walker_noise_idx)`` where
        ``trajectories`` has shape (T_window, N, W).
    """
    with cp.cuda.Device(device_id):
        if lookup_tables_rules_noises.ndim != 4:
            raise ValueError(
                "lookup_tables_rules_noises must have shape (num_rules, num_noises, N, L)"
            )

        num_rules, num_noises, lut_nodes, lut_length = lookup_tables_rules_noises.shape
        if lut_nodes != N:
            raise ValueError("lookup_tables_rules_noises third dimension must equal N")

        max_inputs = node_regulators.shape[1]

        if W <= 0:
            raise ValueError("Number of walkers must be positive")

        if walker_rule_idx.size != W or walker_noise_idx.size != W:
            raise ValueError(
                "walker_rule_idx and walker_noise_idx must have length equal to W"
            )

        if initial_states is None:
            init_states = (cp.random.random((N, W)) <= 0.5).astype(cp.bool_)
        else:
            if initial_states.shape != (N, W):
                raise ValueError("initial_states must have shape (N, W)")
            init_states = initial_states.astype(cp.bool_, copy=False)

        kernel = _get_multi_rule_multi_noise_kernel()
        tables = cp.ascontiguousarray(
            lookup_tables_rules_noises.astype(cp.float32, copy=False)
        )
        regulators = cp.ascontiguousarray(node_regulators.astype(cp.int32, copy=False))
        rule_map = cp.ascontiguousarray(walker_rule_idx.astype(cp.int32, copy=False))
        noise_map = cp.ascontiguousarray(walker_noise_idx.astype(cp.int32, copy=False))

        tpb_nodes, tpb_walkers = threads_per_block
        if tpb_nodes * tpb_walkers > 1024:
            raise ValueError("threads_per_block product must not exceed 1024")
        block = (int(tpb_walkers), int(tpb_nodes))
        grid = (
            (W + tpb_walkers - 1) // tpb_walkers,
            (N + tpb_nodes - 1) // tpb_nodes,
        )

        current = init_states.copy()
        next_state = cp.empty_like(current)

        if T_window is None or T_window > T or T_window < 1:
            T_window = T + 1

        trajectories = cp.empty((T_window, N, W), dtype=cp.bool_)
        trajectories[0, :, :] = current

        for t in range(T):
            mask = maskfunction(
                t,
                N,
                W,
                current,
                threads_per_block=threads_per_block,
            ).astype(cp.float32, copy=False)

            kernel(
                grid,
                block,
                (
                    current,
                    mask,
                    next_state,
                    tables,
                    regulators,
                    rule_map,
                    noise_map,
                    np.int32(N),
                    np.int32(W),
                    np.int32(lut_length),
                    np.int32(max_inputs),
                    np.int32(num_noises),
                ),
            )

            if t >= T - T_window:
                trajectories[t - (T - T_window), :, :] = next_state

            current, next_state = next_state, current

        return trajectories, init_states, rule_map, noise_map


def fitness_split_multi_rule_multi_noise(
    rules: Sequence[str],
    noises: Sequence[float] | None = None,
    n_walkers: int = 1000,
    time_step_factor: int = 5,
    lattice_size: int = 149,
    k: int = 7,
    initial_config_type: str = "normal",
    device_id: int = 0,
    threads_per_block: tuple[int, int] = (32, 32),
    T_window: int = 2,
    stream: cp.cuda.Stream | None = None,
    return_job: bool = False,
) -> dict[str, dict[str, dict[float | str, float]]] | _MultiRuleSimulationJob:
    """Evaluate multiple rules across multiple noise variants in a single simulation.

    This function processes multiple rules and noise levels simultaneously in one
    GPU kernel, providing maximum parallelization efficiency.

    Parameters
    ----------
    rules : Sequence[str]
        List of rule strings to evaluate.
    noises : Sequence[float] | None, optional
        Noise levels to simulate. Defaults to a ten-point sweep if omitted.
    n_walkers : int, optional
        Number of walkers assigned *per rule-noise combination*. The total walkers
        used in the simulation are ``n_walkers * len(rules) * len(noises)``.
    time_step_factor : int, optional
        Multiplier for timesteps: n_time_steps = lattice_size * time_step_factor + 1.
    lattice_size : int, optional
        Size of the CA lattice.
    k : int, optional
        Number of inputs per CA cell.
    initial_config_type : str, optional
        Type of initial configuration: 'normal' or 'uniform_bias_dist'.
    device_id : int, optional
        Identifier of the CuPy device on which to execute the simulation.
    threads_per_block : tuple[int, int], optional
        CUDA block dimensions (walkers, nodes).
    T_window : int, optional
        Number of trailing timesteps to retain.

    Returns
    -------
    dict[str, dict[str, dict[float | str, float]]] or _MultiRuleSimulationJob
        Nested dictionary with structure: {rule: {metric: {noise: value, 'AUC': auc_value}}}
        when ``return_job`` is False. When ``return_job`` is True, returns a job handle
        that can be finalized later to retrieve the metrics, enabling asynchronous
        stream-based execution.
    """

    if noises is None:
        noises = [
            0.000,
            0.010,
            0.020,
            0.030,
            0.040,
            0.050,
            0.060,
            0.070,
            0.080,
            0.090,
            0.100,
        ]

    noises = [float(noise) for noise in noises]

    job = _launch_multi_rule_multi_noise_job(
        rules=rules,
        noises=noises,
        n_walkers=n_walkers,
        time_step_factor=time_step_factor,
        lattice_size=lattice_size,
        k=k,
        initial_config_type=initial_config_type,
        device_id=device_id,
        threads_per_block=threads_per_block,
        T_window=T_window,
        stream=stream,
    )

    if return_job:
        return job

    return _finalize_multi_rule_multi_noise_job(job)


def evaluate_rules_parallel(
    rules: Mapping[str, str],
    parallel_rules: int | None = None,
    *,
    noises: Sequence[float] | None = None,
    n_walkers: int = 1000,
    time_step_factor: int = 5,
    lattice_size: int = 149,
    k: int = 7,
    initial_config_type: str = "normal",
    threads_per_block: tuple[int, int] = (32, 32),
    T_window: int = 2,
    device_id: int | None = None,
) -> tuple[dict[str, dict[str, dict[float | str, float]]], dict[str, float], float]:
    rule_items = list(rules.items())
    if not rule_items:
        return {}, {}, 0.0

    parallel_budget = len(rule_items) if parallel_rules is None else parallel_rules
    parallel_rules = max(1, min(parallel_budget, len(rule_items)))

    noises_payload: Sequence[float] | None
    if noises is None:
        noises_payload = None
    else:
        noises_payload = tuple(float(n) for n in noises)

    results: dict[str, dict[str, dict[float | str, float]]] = {}
    durations: dict[str, float] = {}

    # Helper to map results (keyed by rule string) back to user-provided names,
    # correctly handling duplicate rule strings pointing to different names.
    from collections import defaultdict

    rule_to_names: dict[str, list[str]] = defaultdict(list)
    for name, lut in rule_items:
        rule_to_names[lut].append(name)
    warmup_rule = "".join(["0"] * lattice_size)
    # Warm up GPU to avoid including initialization time in measurements
    fitness_split_multi_rule_multi_noise(
        rules=[warmup_rule],
        noises=noises_payload,
        n_walkers=n_walkers,
        time_step_factor=time_step_factor,
        lattice_size=lattice_size,
        k=k,
        initial_config_type=initial_config_type,
        device_id=device_id if device_id is not None else 0,
        threads_per_block=threads_per_block,
        T_window=T_window,
    )
    wall_start = time.perf_counter()
    # Process in batches (>1) to cap memory usage on GPU using the multi-rule path
    for i in range(0, len(rule_items), parallel_rules):
        batch = rule_items[i : i + parallel_rules]
        batch_names = [name for name, _ in batch]
        batch_rules = [lut for _, lut in batch]

        batch_out = cast(
            dict[str, dict[str, dict[float | str, float]]],
            fitness_split_multi_rule_multi_noise(
                rules=batch_rules,
                noises=noises_payload,
                n_walkers=n_walkers,
                time_step_factor=time_step_factor,
                lattice_size=lattice_size,
                k=k,
                initial_config_type=initial_config_type,
                device_id=device_id if device_id is not None else 0,
                threads_per_block=threads_per_block,
                T_window=T_window,
                return_job=False,
            ),
        )
        # Collect results from this batch; duration attribution will be done globally after all batches
        for rule_str, metrics in batch_out.items():
            # Map to all names that share this rule string
            for name in rule_to_names.get(rule_str, []):
                if name in batch_names:
                    results[name] = metrics

    wall_elapsed = time.perf_counter() - wall_start
    # Attribute average per-rule time based on the total wall time over the full set, not per batch
    total_rules = max(1, len(rule_items))
    avg_per_rule = wall_elapsed / total_rules
    for name, _ in rule_items:
        if name in results:
            durations[name] = avg_per_rule
    return results, durations, wall_elapsed


def sweep_parallel_rule_counts(
    rules: Mapping[str, str],
    parallel_counts: Sequence[int],
    *,
    repeats: int = 1,
    noises: Sequence[float] | None = None,
    n_walkers: int = 1000,
    time_step_factor: int = 5,
    lattice_size: int = 149,
    k: int = 7,
    initial_config_type: str = "normal",
    threads_per_block: tuple[int, int] = (32, 32),
    T_window: int = 2,
    device_id: int | None = None,
) -> list[dict[str, float]]:
    num_rules = len(rules)
    if num_rules == 0:
        return []

    noise_count = len(noises) if noises is not None else len(DEFAULT_NOISES)
    walkers_per_noise = float(n_walkers)
    total_walkers = float(n_walkers * noise_count)

    sweep_results: list[dict[str, float]] = []
    for parallel_rules in parallel_counts:
        print(f"n_walkers: {n_walkers}, parallel_rules: {parallel_rules}")
        per_rule_times: list[float] = []
        wall_times: list[float] = []
        throughputs: list[float] = []
        for _ in range(max(1, repeats)):
            _, durations, wall_elapsed = evaluate_rules_parallel(
                rules,
                parallel_rules,
                noises=noises,
                n_walkers=n_walkers,
                time_step_factor=time_step_factor,
                lattice_size=lattice_size,
                k=k,
                initial_config_type=initial_config_type,
                threads_per_block=threads_per_block,
                T_window=T_window,
                device_id=device_id,
            )
            if durations:
                wall_times.append(wall_elapsed)
                per_rule_time = wall_elapsed / num_rules if num_rules else 0.0
                per_rule_times.append(per_rule_time)
                if wall_elapsed > 0:
                    throughputs.append(num_rules / wall_elapsed)
                else:
                    throughputs.append(0.0)
        if wall_times:
            sweep_results.append({
                "parallel_rules": float(parallel_rules),
                "total_rules": float(num_rules),
                "walkers_per_noise": walkers_per_noise,
                "total_walkers": total_walkers,
                "total_wall_time": float(np.sum(wall_times)),
                "avg_wall_time": float(np.mean(wall_times)),
                "std_wall_time": float(np.std(wall_times))
                if len(wall_times) > 1
                else 0.0,
                "avg_time_per_rule": float(np.mean(per_rule_times))
                if per_rule_times
                else 0.0,
                "std_time_per_rule": float(np.std(per_rule_times))
                if len(per_rule_times) > 1
                else 0.0,
                "avg_rules_per_second": float(np.mean(throughputs))
                if throughputs
                else 0.0,
                "std_rules_per_second": float(np.std(throughputs))
                if len(throughputs) > 1
                else 0.0,
            })
    return sweep_results


def save_sweep_results_to_csv(
    results: Sequence[dict[str, float]],
    output_path: str | Path,
) -> None:
    if not results:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in results for key in row.keys()})
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def plot_sweep_results(
    results: Sequence[dict[str, float]],
    *,
    title: str = "Parallel Rule Sweep",
    show: bool = False,
    save_path: str | Path | None = None,
) -> None:
    if not results:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    walker_groups = sorted({
        walkers
        for row in results
        for walkers in [row.get("walkers_per_noise")]
        if walkers is not None
    })
    if len(walker_groups) <= 1:
        counts = [row.get("parallel_rules", 0.0) for row in results]
        avg_times = [row.get("avg_time_per_rule", 0.0) for row in results]
        wall_times = [row.get("avg_wall_time", 0.0) for row in results]
        std_times = [row.get("std_time_per_rule", 0.0) for row in results]
        label = (
            f"Avg time/rule ({int(walker_groups[0])} walkers/noise)"
            if walker_groups and walker_groups[0]
            else "Avg time/rule"
        )
        plt.errorbar(counts, avg_times, yerr=std_times, fmt="o-", label=label)
        plt.plot(counts, wall_times, "s-", label="Avg wall time")
    else:
        for walkers in walker_groups:
            group_rows = [
                row for row in results if row.get("walkers_per_noise") == walkers
            ]
            group_rows.sort(key=lambda row: row.get("parallel_rules", 0.0))
            counts = [row.get("parallel_rules", 0.0) for row in group_rows]
            avg_times = [row.get("avg_time_per_rule", 0.0) for row in group_rows]
            std_times = [row.get("std_time_per_rule", 0.0) for row in group_rows]
            label = (
                f"Avg time/rule ({int(walkers)} walkers/noise)"
                if walkers
                else "Avg time/rule"
            )
            plt.errorbar(counts, avg_times, yerr=std_times, fmt="o-", label=label)

    plt.xlabel("Parallel rules")
    plt.ylabel("Seconds")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    automata_output_list = {
        "GKL": "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011111",
        "GP": "00000101000000000101010100000101000001010000000001010101000001010101010111111111010101011111111101010101111111110101010111111111",
        "GEP_1": "00010001000000000101010100000000000100010000111101010101000011110001000111111111010101011111111100010001111111110101010111111111",
        "GEP_2": "00000000010101010000000001110111000000000101010100000000011101110000111101010101000011110111011111111111010101011111111101110111",
        "Das": "00000111000000000000011111111111000011110000000000001111111111110000111100000000000001111111111100001111001100010000111111111111",
        "Davis": "00000000001011110000001101011111000000000001111111001111000111110000000000101111111111000101111100000000000111111111111100011111",
        "DMC": "00000101000001000000010110000111000001010000000000001111011101110000001101110111010101011000001101111011111111111011011101111111",
        "COE_1": "00000001000101000011000011010111000100010000111100111001010101110000010110110100111111110001011111110001001111011111100101010111",
        "COE_2": "00010100010100010011000001011100000000000101000011001110010111110001011100010001111111110101111100001111010100111100111101011111",
        "MM401": "00010101000000000101010100000000000101010000000001010101000011110001010111111111010101011111111100010101111111110101010111111111",
        "MM802": "00010100010100010000000011011100000011110001000000001110010111110001011100010001111111111101111100001111000100111100111101011111",
        "F_WO_1": "00010100011000010000011101111100000011110000000011001110101111110001011100100001000001000111111100001111001100110011111110111111",
        "F_WO_2": "00000010000000110011001100001111000000011101111101111011000101110000001010001100111111110000111111000001000111110111100111010111",
        "F_WO_3": "00000011010100000100010000011111000000011100111111111000000011110000001101011100111101110001111100000001001111111111101100011111",
        "F_WO_4": "00000111001000000000001101111111000001110001000011000101001111110000111111100000000011000111111100000111110111011111010100111111",
        "F_WO_5": "00010000010000010100001100011100000000011101111111110000000011110001001101001101111111110001111100000001110111111111001100011111",
        "F_WO_10": "00000111001000000000001101111111000001110001000011001101001111110000111111100000000011000111111100000111000111011111110100111111",
        "F_WO_50": "00000000010111110100000000010111000000001111011100011000010001110000000001011111111100111101011111111100001101111101101101010111",
        "F_WO_100": "00000010000000110011001100001111000000011101011101010011000101110000001010001100111111111100111111001101110101110101000111010111",
        "F_WO_500": "00000111011000100100001101110011000101110001000011000100001011110000010001100010000011000111111100010111110111111111011100111111",
        "F_WO_1000": "00000111011000000100001101111111000001110001000001000100001001110000011101100000000011000111111111000111110111110111011111110111",
        "F_WO_3000": "00010001001100000111001010100000000101110101001100110011010011110001000100000011110011101010111100010111010111111100111101011111",
        "F_WO_7000": "00010000000100110000010000011111000000011100111101011001000101110001101011011100111101110001111111000101001111010101100111010111",
    }
    # for name, rule in automata_output_list.items():
    #     start_time = time.time()
    #     accuracy = fitness_split_multi_variant(rule=rule,n_walkers = 1000)
    #     end_time = time.time()
    #     # print(f"Results for {name}:")
    #     # for noise_level, metrics in results.items():
    #     #     print(f"  Noise {noise_level:.3f}: {metrics}")
    #     print(f"Name: {name}, score: {accuracy['AUC']['minP0P1']}")
    #     print(f"Evaluation time for {name}: {end_time - start_time:.2f} seconds\n")
    #     gc.collect()
    random_rules = {
        f"Random_{i}": "".join(random.choice("01") for _ in range(128))
        for i in range(80)
    }
    sweep_stats = []
    for n_walkers in [
        1000,
        10000,
    ]:
        # sample up to 80 distinct rules from the full set; if fewer than 80 exist, take all
        sweep_counts = [
            1,
            2,
            5,
            10,
            20,
            40,
        ]
        sample_size = max(sweep_counts)
        sampled_items = random.sample(list(random_rules.items()), k=sample_size)
        subset = dict(sampled_items)
        stats = sweep_parallel_rule_counts(
            subset,
            sweep_counts,
            repeats=1,
            n_walkers=n_walkers,
            noises=None,
        )
        sweep_stats.extend(stats)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    csv_path = artifacts_dir / "parallel_multi_rule_sweep.csv"
    save_sweep_results_to_csv(sweep_stats, csv_path)
    plot_path = artifacts_dir / "parallel_multi_rule_sweep.png"
    plot_sweep_results(sweep_stats, save_path=plot_path)
    for stat in sweep_stats:
        avg_rules_per_second = stat.get("avg_rules_per_second", 0.0)
        walkers_per_noise = stat.get("walkers_per_noise", 0.0)
        total_wall_time = stat.get("total_wall_time", 0.0)
        print(
            f"Parallel rules: {int(stat['parallel_rules'])}, avg time/rule: {stat['avg_time_per_rule']:.3f}s, "
            f"wall: {stat['avg_wall_time']:.3f}s, total: {total_wall_time:.3f}s, rules/s: {avg_rules_per_second:.2f}, "
            f"walkers/noise: {int(walkers_per_noise)}"
        )
    gc.collect()
