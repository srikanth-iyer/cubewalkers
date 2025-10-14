import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import cupy as cp
import cubewalkers as cw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from fitness import ca_network_from_output_list, compute_accuracy


def _dbg_header(title: str) -> None:
    print(f"\n\n=== {title} ===\n")


def _summarize_array(
    name: str, arr: Any, max_rows: int = 50, max_cols: int = 50
) -> None:
    """Print a compact debug summary of an array-like object.

    This helper writes to stdout a small, safe-to-compute summary intended for
    debugging GPU/CPU arrays without moving large data to host memory.

    It prints:
    - The object type, dtype, and shape
    - A small slice/sample (up to max_rows x max_cols) converting CuPy arrays
      to NumPy as needed
    - Lightweight stats (for bool: true count; for numeric: min/max/mean),
      guarded by try/except to avoid expensive or failing operations

    Parameters
    - name: Label used in the output for the array
    - arr: Array-like object (CuPy ndarray, NumPy ndarray, or similar)
    - max_rows: Max rows to include in the printed sample
    - max_cols: Max columns to include in the printed sample

    Notes
    - This function is side-effect-only (prints) and returns None.
    - It attempts to minimize device-to-host transfers by slicing before
      converting CuPy arrays to NumPy.
    - All operations are wrapped in try/except so it’s safe in most contexts.
    """
    try:
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        print(f"{name}: type={type(arr).__name__}, dtype={dtype}, shape={shape}")
        if hasattr(arr, "ndim") and shape is not None:
            # Print a small sample to avoid large transfers
            try:
                if isinstance(arr, cp.ndarray):
                    dims = (max_rows, max_cols)
                    extra = max(0, int(getattr(arr, "ndim", 0)) - 2)
                    dims = dims + (max_cols,) * extra
                    sl = tuple(slice(0, min(s, m)) for s, m in zip(shape, dims))
                    sample = cp.asnumpy(arr[sl])
                else:
                    dims = (max_rows, max_cols)
                    extra = max(0, int(getattr(arr, "ndim", 0)) - 2)
                    dims = dims + (max_cols,) * extra
                    sl = tuple(slice(0, min(s, m)) for s, m in zip(shape, dims))
                    sample = arr[sl]
                print(f"{name} sample[0:{max_rows},0:{max_cols}]:\n{sample}")
            except Exception as e:
                print(f"{name} sample unavailable: {e}")

            # Lightweight stats
            try:
                if isinstance(arr, cp.ndarray):
                    if arr.dtype == cp.bool_:
                        trues = int(cp.sum(arr).item())
                        print(f"{name} stats: true_count={trues}")
                    elif arr.dtype is not None and arr.dtype.kind in ("f", "i", "u"):
                        mn = float(cp.min(arr).item())
                        mx = float(cp.max(arr).item())
                        mean = float(cp.mean(arr).item())
                        print(
                            f"{name} stats: min={mn:.4g}, max={mx:.4g}, mean={mean:.4g}"
                        )
                else:
                    if getattr(arr, "dtype", None) == np.bool_:
                        trues = int(arr.sum())
                        print(f"{name} stats: true_count={trues}")
                    elif getattr(arr, "dtype", None) is not None and arr.dtype.kind in (
                        "f",
                        "i",
                        "u",
                    ):
                        mn = float(arr.min())
                        mx = float(arr.max())
                        mean = float(arr.mean())
                        print(
                            f"{name} stats: min={mn:.4g}, max={mx:.4g}, mean={mean:.4g}"
                        )
            except Exception as e:
                print(f"{name} stats unavailable: {e}")
    except Exception as e:
        print(f"Failed to summarize {name}: {e}")


def _first_n(seq: Any, n: int = 3) -> Any:
    try:
        return list(seq)[:n]
    except Exception:
        try:
            return tuple(seq)[:n]
        except Exception:
            return f"<unprintable type {type(seq).__name__}>"


def compute_minP0P1_auc(
    rule_lut: str,
    probabilities: Iterable[float],
    n_walkers: int = 10000,
    k: int = 7,
    lattice_size: int = 149,
    n_time_steps: int | None = None,
    t_window: int = 2,
    rng_seed: int | None = None,
) -> float:
    """Compute the AUC of minP0P1 over a probability sweep for a CA rule.

    Returns the trapezoidal AUC integrating minP0P1 vs probability.
    """
    import numpy as np

    # Build CANA network from rule LUT
    net = ca_network_from_output_list(rule_lut, k=k, lattice=lattice_size)

    # Deterministic LUT once
    outs, ins = cw.conversions.cana2cupyLUT(net)
    outs_f = outs.astype(cp.float32, copy=False)

    # CuPy array of probabilities (ensure 1-D)
    probs_cp = cp.asarray(list(probabilities), dtype=cp.float32)
    if probs_cp.ndim == 0:
        probs_cp = probs_cp.reshape(1)

    # Batch construct probabilistic LUTs on device: p + outs*(1-2p)
    prob_luts = (
        probs_cp[:, None, None]
        + outs_f[None, ...] * (1.0 - 2.0 * probs_cp)[:, None, None]
    )

    # Default steps if not provided
    if n_time_steps is None:
        n_time_steps = lattice_size * 5 + 1

    # Create probabilistic model once; swap LUTs per probability
    model = cw.Model(
        lookup_tables=prob_luts[0],
        node_regulators=ins,
        n_time_steps=n_time_steps,
        n_walkers=n_walkers,
        probabilistic_lut=True,
    )

    # Sweep
    minP0P1_vals: list[float] = []
    for i in range(int(probs_cp.shape[0])):
        if rng_seed is not None:
            cp.random.seed(int(rng_seed))
        model.lookup_tables = prob_luts[i]
        model.simulate_ensemble(
            maskfunction=cw.update_schemes.synchronous_PBN,
            T_window=t_window,
            averages_only=False,
        )
        p_val = float(probs_cp[i].item())
        acc = compute_accuracy(model, noise=p_val)
        val = acc.get("minP0P1") if isinstance(acc, dict) else None
        minP0P1_vals.append(float(val) if val is not None else np.nan)

    x = np.asarray(cp.asnumpy(probs_cp), dtype=float)
    y = np.asarray(minP0P1_vals, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(y[mask], x[mask]))


def compute_minP0P1_auc_for_rules(
    rule_luts: Iterable[str],
    probabilities: Iterable[float],
    n_walkers: int = 1000,
    k: int = 7,
    lattice_size: int = 149,
    n_time_steps: int | None = None,
    t_window: int = 2,
    rng_seed: int | None = None,
    num_processes: int = 1,
) -> dict[str, float]:
    """Compute AUC(minP0P1) for multiple CA rules efficiently.

    This implementation evaluates rules in parallel using separate processes when
    ``num_processes`` > 1, avoiding CUDA stream coordination. Each process
    constructs its own probabilistic model for the assigned rule while reusing a
    shared set of initial conditions supplied by the parent process to preserve
    fairness across rules.
    """

    rules = list(rule_luts)
    if len(rules) == 0:
        return {}

    probability_list: tuple[float, ...] = tuple(float(p) for p in probabilities)
    if len(probability_list) == 0:
        return {rule: float("nan") for rule in rules}

    if n_time_steps is None:
        n_time_steps = lattice_size * 5 + 1

    rng = np.random.default_rng(rng_seed)
    base_initial_states = np.ascontiguousarray(
        (rng.random((lattice_size, n_walkers)) < 0.5).astype(np.bool_, copy=False)
    )

    shared_args = (
        probability_list,
        n_walkers,
        k,
        lattice_size,
        n_time_steps,
        t_window,
        base_initial_states,
    )

    if num_processes <= 1:
        results = {}
        for rule in rules:
            _, auc = _minp0p1_process_worker((rule, *shared_args))
            results[rule] = auc
        return results

    max_workers = max(1, min(int(num_processes), len(rules)))
    worker_args = [(rule, *shared_args) for rule in rules]
    results: dict[str, float] = {}
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
        for rule, auc in executor.map(_minp0p1_process_worker, worker_args):
            results[rule] = auc
    return results


def _minp0p1_process_worker(
    args: tuple[
        str,
        Sequence[float],
        int,
        int,
        int,
        int,
        int,
        np.ndarray,
    ]
) -> tuple[str, float]:
    (
        rule,
        probabilities,
        n_walkers,
        k,
        lattice_size,
        n_time_steps,
        t_window,
        initial_states_np,
    ) = args

    import numpy as _np

    deviations = 3.0

    with cp.cuda.Device():
        probs_cp = cp.asarray(probabilities, dtype=cp.float32)
        if probs_cp.ndim == 0:
            probs_cp = probs_cp.reshape(1)

        if probs_cp.size == 0:
            return rule, float("nan")

        net = ca_network_from_output_list(rule, k=k, lattice=lattice_size)
        outs, ins = cw.conversions.cana2cupyLUT(net)
        outs_f = outs.astype(cp.float32, copy=False)

        lut_buf = cp.empty_like(outs_f, dtype=cp.float32)
        model = cw.Model(
            lookup_tables=lut_buf,
            node_regulators=ins,
            n_time_steps=n_time_steps,
            n_walkers=n_walkers,
            probabilistic_lut=True,
        )

        init_states_cp = cp.asarray(initial_states_np, dtype=cp.bool_)
        init_sums = cp.sum(init_states_cp, axis=0)
        midpoint = lattice_size // 2 + 1
        start_above = init_sums >= midpoint
        start_below = init_sums < midpoint
        denom_above = int(cp.sum(start_above).item())
        denom_below = int(cp.sum(start_below).item())

        min_vals: list[float] = []
        for idx in range(int(probs_cp.shape[0])):
            p_val = float(probs_cp[idx].item())
            cp.multiply(outs_f, (1.0 - 2.0 * p_val), out=lut_buf)
            cp.add(lut_buf, p_val, out=lut_buf)
            model.lookup_tables = lut_buf
            model.initial_states = init_states_cp.copy()
            model.simulate_ensemble(
                maskfunction=cw.update_schemes.synchronous_PBN,
                T_window=t_window,
                averages_only=False,
            )
            traj = model.trajectories
            if traj.shape[0] < 2:
                min_vals.append(float("nan"))
                continue

            eps = (p_val * lattice_size + (p_val * (1.0 - p_val) * lattice_size) ** 0.5 * deviations) // 1
            thr_hi = lattice_size - eps
            thr_lo = eps

            last2 = traj[-2:]
            sums_last2 = cp.sum(last2, axis=1)
            end_above = (sums_last2[1] >= thr_hi) & (sums_last2[0] >= thr_hi)
            end_below = (sums_last2[1] <= thr_lo) & (sums_last2[0] <= thr_lo)

            tp = float(cp.sum(start_above & end_above).item())
            tn = float(cp.sum(start_below & end_below).item())

            p1 = tp / denom_above if denom_above > 0 else 0.0
            p0v = tn / denom_below if denom_below > 0 else 0.0
            min_vals.append(min(p0v, p1))

        x = _np.asarray(probabilities, dtype=float)
        y = _np.asarray(min_vals, dtype=float)
        mask = ~_np.isnan(y)
        if mask.sum() < 2:
            return rule, float("nan")
        return rule, float(_np.trapezoid(y[mask], x[mask]))


if __name__ == "__main__":
    # _dbg_header("Runtime and environment")
    # print(f"Python: {sys.version.split()[0]}")
    # print(f"CuPy: {cp.__version__}")
    # try:
    #     ndev = cp.cuda.runtime.getDeviceCount()
    #     print(f"CUDA devices: {ndev}")
    #     if ndev > 0:
    #         dev = cp.cuda.Device()
    #         props = cp.cuda.runtime.getDeviceProperties(dev.id)
    #         name = props.get("name", b"")
    #         if isinstance(name, (bytes, bytearray)):
    #             name = name.decode(errors="ignore")
    #         print(f"Active device: id={dev.id}, name={name}")
    # except Exception as e:
    #     print(f"CUDA device info unavailable: {e}")
    rule1 = "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011111"  # gkl
    rule2 = "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011110"
    rule3 = "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011101"
    # rule = "0111111001011010"

    # logic = {
    #     0: {
    #         "name": "0",
    #         "in": [2, 0, 1],
    #         "out": ["0", "0", "0", "1", "0", "0", "0", "0"],
    #     },
    #     1: {
    #         "name": "1",
    #         "in": [0, 1, 2],
    #         "out": ["0", "1", "0", "1", "1", "0", "1", "0"],
    #     },
    #     2: {
    #         "name": "2",
    #         "in": [1, 2, 0],
    #         "out": ["1", "0", "0", "1", "0", "0", "1", "0"],
    #     },
    # }
    # initial_states = cp.array([
    #     [True, False, True, False, False],
    #     [True, False, True, False, False],
    #     [True, False, True, False, False],
    # ])
    # ==========================================================================

    # _dbg_header("CANA network from logic")
    # print(f"logic nodes: {len(logic)}; keys={list(logic.keys())}")

    # # net = BooleanNetwork.from_dict(logic)
    # net = ca_network_from_output_list(rule, k=7, lattice=149)

    # prob = 0.01

    # # ==========================================================================

    # _dbg_header("Generating lookup tables from CANA")
    # t0 = time.perf_counter()
    # prob_outs, prob_ins = cw.conversions.cana2cupy_probabilisticLUT(net, prob)
    # t1 = time.perf_counter()
    # outs, ins = cw.conversions.cana2cupyLUT(net)
    # t2 = time.perf_counter()
    # print(
    #     f"cana2cupy_probabilisticLUT time: {(t1 - t0) * 1e3:.2f} ms; cana2cupyLUT time: {(t2 - t1) * 1e3:.2f} ms"
    # )
    # _summarize_array("prob_outs", prob_outs, max_rows=6, max_cols=12)
    # print(
    #     f"prob_ins (regulators): len={len(prob_ins)}; first={prob_ins[0] if len(prob_ins) > 0 else None}"
    # )
    # _summarize_array("outs", outs, max_rows=6, max_cols=12)
    # print(f"ins (regulators): len={len(ins)}; first={ins[0] if len(ins) > 0 else None}")
    # try:
    #     if isinstance(prob_outs, cp.ndarray):
    #         if not cp.logical_and(prob_outs >= 0.0, prob_outs <= 1.0).all():
    #             print("WARNING: prob_outs values outside [0,1]")
    #     if isinstance(outs, cp.ndarray) and outs.dtype != cp.bool_:
    #         print(f"WARNING: outs dtype expected bool, got {outs.dtype}")
    # except Exception as e:
    #     print(f"LUT sanity checks failed: {e}")

    # prob_test_model = cw.Model(
    #     lookup_tables=prob_outs,
    #     node_regulators=prob_ins,
    #     n_time_steps=100,
    #     n_walkers=5000,
    #     probabilistic_lut=True,
    # )
    # test_model = cw.Model(
    #     lookup_tables=outs,
    #     node_regulators=ins,
    #     n_time_steps=100,
    #     n_walkers=5000,
    #     probabilistic_lut=False,
    # )

    # # ==========================================================================
    # _dbg_header("Models created")
    # print(
    #     f"prob_test_model: name={prob_test_model.name}, n_vars={prob_test_model.n_variables}, n_walkers={prob_test_model.n_walkers}, n_time_steps={prob_test_model.n_time_steps}, probabilistic={prob_test_model.probabilistic_lut}"
    # )
    # print(
    #     f"test_model:      name={test_model.name}, n_vars={test_model.n_variables}, n_walkers={test_model.n_walkers}, n_time_steps={test_model.n_time_steps}, probabilistic={test_model.probabilistic_lut}"
    # )
    # _summarize_array(
    #     "prob_test_model.lookup_tables",
    #     prob_test_model.lookup_tables,
    #     max_rows=6,
    #     max_cols=12,
    # )
    # _summarize_array(
    #     "test_model.lookup_tables", test_model.lookup_tables, max_rows=6, max_cols=12
    # )
    # print(
    #     f"prob_test_model.node_regulators[0:3] (sample)={_first_n(prob_test_model.node_regulators, 3)}"
    # )
    # print(
    #     f"test_model.node_regulators[0:3] (sample)={_first_n(test_model.node_regulators, 3)}"
    # )

    # # ==========================================================================

    # # _dbg_header("Initial states")
    # # _summarize_array("initial_states", initial_states)
    # # try:
    # #     if (
    # #         initial_states.shape[0] != prob_test_model.n_variables
    # #         or initial_states.shape[1] != prob_test_model.n_walkers
    # #     ):
    # #         print(
    # #             f"WARNING: initial_states shape {initial_states.shape} != (n_variables={prob_test_model.n_variables}, n_walkers={prob_test_model.n_walkers})"
    # #         )
    # # except Exception as e:
    # #     print(f"Initial states shape check failed: {e}")

    # # prob_test_model.initial_states = initial_states
    # # test_model.initial_states = initial_states
    # # ==========================================================================

    # _dbg_header("Simulate ensemble (probabilistic)")
    # try:
    #     t0 = time.perf_counter()
    #     prob_test_model.simulate_ensemble(
    #         maskfunction=cw.update_schemes.synchronous_PBN,
    #         T_window=20,
    #         averages_only=False,
    #     )
    #     t1 = time.perf_counter()
    #     print(f"simulate_ensemble (prob) time: {(t1 - t0):.3f} s")
    #     # _summarize_array("prob_test_model.trajectories", prob_test_model.trajectories)
    # except Exception as e:
    #     print(f"ERROR during probabilistic simulation: {e}")
    # # ==========================================================================

    # _dbg_header("Simulate ensemble (deterministic)")
    # try:
    #     t0 = time.perf_counter()
    #     test_model.simulate_ensemble(
    #         maskfunction=cw.update_schemes.synchronous_PBN,
    #         T_window=20,
    #         averages_only=False,
    #     )
    #     t1 = time.perf_counter()
    #     print(f"simulate_ensemble (det) time: {(t1 - t0):.3f} s")
    #     # _summarize_array("test_model.trajectories", test_model.trajectories)
    # except Exception as e:
    #     print(f"ERROR during deterministic simulation: {e}")
    # # ==========================================================================

    # _dbg_header("Post-run quick checks")
    # try:
    #     print("sum(prob trajectories):", cp.sum(prob_test_model.trajectories).item())
    #     print("sum(det trajectories):", cp.sum(test_model.trajectories).item())
    #     try:
    #         p_mean = float(cp.mean(prob_test_model.trajectories).item())
    #         d_mean = float(cp.mean(test_model.trajectories).item())
    #         print(f"mean(prob traj)={p_mean:.4g}, mean(det traj)={d_mean:.4g}")
    #     except Exception:
    #         pass
    # except Exception as e:
    #     print(f"Post-run checks failed: {e}")
    # # not sure how to test this, because the output varies even in non probabilistic LUTs where the array is boolean. can you help me with this?

    # ----------------------------------------------------------------------
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
    rules = list(automata_output_list.values())
    _dbg_header("Batch-generate probabilistic LUTs and sweep probabilities (fast path)")
    # Vectorized construction of many probabilistic LUTs on-GPU from the single boolean LUT.
    # Given outs in {0,1}, the probabilistic value is p for 0 and 1-p for 1:
    # prob_LUT = p + outs * (1 - 2p)
    probabilities = np.linspace(0.0, 0.1, num=11, dtype=float)
    run_count = 5
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_records: list[dict[str, float | int]] = []
    for n_walkers in [
        10000,
        # 5000, 
        1000, 
        # 100, 
        ]:
        for num_processes in range(5, 0, -1):
            print(
                f"\n\n=== Sweep with n_walkers={n_walkers}, num_processes={num_processes} ==="
            )
            for number_of_rules in range(6, 0, -1):
                print(f"\n\n--- Sweep {number_of_rules} rules, {num_processes} processes, {n_walkers} walkers ---")
                run_times: list[float] = []
                run_mean_aucs: list[float] = []
                for run_idx in range(1, run_count + 1):
                    try:
                        rules_list = random.sample(rules, k=number_of_rules)
                        t0 = time.perf_counter()
                        aucs = compute_minP0P1_auc_for_rules(
                            rule_luts=rules_list,
                            probabilities=probabilities,
                            n_walkers=n_walkers,
                            k=7,
                            lattice_size=149,
                            n_time_steps=149 * 5,
                            num_processes=num_processes,
                        )
                        t1 = time.perf_counter()
                        duration = t1 - t0
                        mean_auc = (
                            float(np.nanmean(list(aucs.values())))
                            if aucs
                            else float("nan")
                        )
                        run_times.append(duration)
                        run_mean_aucs.append(mean_auc)
                        if np.isnan(mean_auc):
                            print(
                                f"Run {run_idx}: total time {duration:.3f} s (processes={num_processes})"
                            )
                        else:
                            print(
                                "Run {idx}: total time {dur:.3f} s, mean minP0P1 AUC "
                                "{auc:.6f} (processes={processes})".format(
                                    idx=run_idx,
                                    dur=duration,
                                    auc=mean_auc,
                                    processes=num_processes,
                                )
                            )
                    except Exception as e:
                        print(f"Run {run_idx} failed: {e}")
                if not run_times:
                    print("All runs failed; skipping statistics for this sweep.")
                    continue
                avg_time = float(np.mean(run_times))
                std_time = (
                    float(np.std(run_times, ddof=1)) if len(run_times) > 1 else 0.0
                )
                avg_per_rule = avg_time / number_of_rules
                avg_auc = (
                    float(np.nanmean(run_mean_aucs)) if run_mean_aucs else float("nan")
                )
                print(
                    f"Average total time over {len(run_times)} runs: {avg_time:.3f} s"
                )
                print(f"Average time per rule: {avg_per_rule:.3f} s")
                if not np.isnan(avg_auc):
                    print(f"Average minP0P1 AUC across runs: {avg_auc:.6f}")
                sweep_records.append({
                    "number_of_rules": number_of_rules,
                    "n_walkers": n_walkers,
                    "num_processes": num_processes,
                    "avg_total_time_s": avg_time,
                    "std_total_time_s": std_time,
                    "avg_time_per_rule_s": avg_per_rule,
                    "successful_runs": len(run_times),
                    "total_runs": run_count,
                })

    csv_path = output_dir / "multi_rule_processes_sweep_stats.csv"
    plot_path = output_dir / "multi_rule_processes_sweep_avg_times.png"

    if not sweep_records:
        print("No sweep statistics collected; skipping CSV and plot generation.")
    else:
        stats_df = pd.DataFrame(sweep_records)
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved sweep statistics to {csv_path}")

        sns.set(style="whitegrid")
        stats_df["number_of_rules"] = stats_df["number_of_rules"].astype(int)
        stats_df["n_walkers"] = stats_df["n_walkers"].astype(int)
        stats_df["num_processes"] = stats_df["num_processes"].astype(int)
        stats_df["avg_time_per_rule_s"] = stats_df["avg_time_per_rule_s"].astype(float)

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=stats_df,
            x="number_of_rules",
            y="avg_time_per_rule_s",
            hue="num_processes",
            style="n_walkers",
            markers=True,
            palette="tab10",
            estimator="mean",
            ci=None,
        )
        ax.set_xlabel("Number of rules")
        ax.set_ylabel("Average time per rule (s)")
        ax.set_title(
            "Average time per rule vs Number of rules\nHue = processes, style = walkers"
        )
        ax.legend(title="processes (hue) / Walkers (style)", loc="best")
        ax.set_xticks(sorted(stats_df["number_of_rules"].unique()))
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"Saved sweep timing plot to {plot_path}")
