from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
import gc
import cupy as cp
import numpy as np
import pandas as pd
import cubewalkers as cw
from cana.boolean_network import BooleanNetwork
from scipy import integrate
from sklearn.metrics import matthews_corrcoef


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


def fitness_split(
    rule: str,
    noise: float = 0.0,
    n_walkers: int = 10000,
    time_step_factor: int = 5,  # changing n_walkers to 10000
    initial_config_type="normal",
    lattice_size=149,
    k=7,
) -> dict:
    deviations = 3
    output_list = list(rule)
    network = ca_network_from_output_list(output_list, k=k, lattice=lattice_size)
    outs, ins = cw.conversions.cana2cupy_probabilisticLUT(network, prob=noise)
    model = cw.Model(
        lookup_tables=outs,
        node_regulators=ins,
        n_time_steps=lattice_size * time_step_factor + 1,
        n_walkers=n_walkers,
        probabilistic_lut=True,
    )
    if initial_config_type not in ["normal", "uniform_bias_dist"]:
        raise ValueError(
            "initial_config_type must be either 'normal' or 'uniform_bias_dist'"
        )
    if initial_config_type == "uniform_bias_dist":
        # Generate biases uniformly distributed from 0 to 1
        biases = cp.linspace(0, 1, n_walkers)
        # Initialize the states array
        initial_states = cp.random.random((lattice_size, n_walkers)) <= biases
        # Update: ensure boolean type
        model.initial_states = initial_states.astype(cp.bool_)
    else:
        model.initialize_walkers()

    model.simulate_ensemble(
        maskfunction=cw.update_schemes.synchronous_PBN, T_window=2
    )  # storing last two time-steps for convergence check. Single timestep doesn't spot alternating convergence.
    accuracies = compute_accuracy(model, noise=noise, deviations=deviations)

    return accuracies


def fitness_split_auc(
    rule: str,
    n_walkers: int = 10000,  # changing n_walkers to 10000
    time_step_factor: int = 5,
    device_id=0,
    initial_config_type="normal",
    noises=None,
    lattice_size=149,
    k=7,
    parallel: bool = True,
    max_workers: int | None = None,
    debug: bool = False,
):
    """
    Function that computes the fitness of a rule

    Parameters
    ----------
    rule: The rule to evaluate
    noise: The noise level
    n_walkers: The number of simultaneous calculations
    time_step_factor: timesteps = lattice size * time_step_factor + 1

    Returns
    -------
    float: The fitness of the rule
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

    # Initialize accuracy dictionary with a factory function
    metrics = [
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
    accuracy_dict = {metric: {} for metric in metrics}
    accuracy_dict["AUC"] = {}

    def _run_single(noise: float) -> tuple[float, dict, float]:
        # Per-noise worker: bind device, use a dedicated CUDA stream to enable concurrency
        with cp.cuda.Device(device_id):
            stream = cp.cuda.Stream(non_blocking=True)
            # Use a distinct RNG seed per worker to avoid correlation
            cp.random.seed(int((noise * 1e6) % (2**32 - 1)))
            with stream:
                # NVTX label for timeline (optional)
                if debug:
                    try:
                        from cupy.cuda import nvtx  # type: ignore

                        _push = getattr(nvtx, "RangePush", None)
                        if callable(_push):
                            _push(f"noise={noise}")
                    except Exception:
                        pass
                start_evt = cp.cuda.Event()
                end_evt = cp.cuda.Event()
                start_evt.record(stream)
                res = fitness_split(
                    rule=rule,
                    noise=noise,
                    n_walkers=n_walkers,
                    time_step_factor=time_step_factor,
                    initial_config_type=initial_config_type,
                    lattice_size=lattice_size,
                    k=k,
                )
                end_evt.record(stream)
                if debug:
                    try:
                        from cupy.cuda import nvtx  # type: ignore

                        _pop = getattr(nvtx, "RangePop", None)
                        if callable(_pop):
                            _pop()
                    except Exception:
                        pass
            stream.synchronize()
            # Use CuPy's module-level API to compute elapsed time between events
            elapsed_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
            return noise, res, float(elapsed_ms)

    # Execute either in parallel (CUDA streams) or sequentially as fallback
    wall_start = time.perf_counter()
    per_noise_ms: dict[float, float] = {}
    try:
        if parallel and len(noises) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for noise, fitnesses, t_ms in executor.map(_run_single, noises):
                    for state, value in fitnesses.items():
                        accuracy_dict[state][noise] = value
                    per_noise_ms[noise] = t_ms
        else:
            # Sequential path
            for noise in noises:
                _, fitnesses, t_ms = _run_single(noise)
                for state, value in fitnesses.items():
                    accuracy_dict[state][noise] = value
                per_noise_ms[noise] = t_ms

    except cp.cuda.runtime.CUDARuntimeError as e:
        # Fallback to sequential on GPU runtime errors
        print(f"GPU parallel error; falling back to sequential. Detail: {e}")
        for noise in noises:
            _, fitnesses, t_ms = _run_single(noise)
            for state, value in fitnesses.items():
                accuracy_dict[state][noise] = value
            per_noise_ms[noise] = t_ms
    wall_elapsed = time.perf_counter() - wall_start

    if debug:
        try:
            total_ms = sum(per_noise_ms.values())
            print("\n[fitness_split_auc] Debug timings:")
            print(f"- noises: {list(noises)}")
            print(
                f"- per-noise GPU times (ms): { {round(k, 3): round(v, 2) for k, v in per_noise_ms.items()} }"
            )
            print(
                f"- sum per-noise: {total_ms:.2f} ms; wall: {wall_elapsed * 1e3:.2f} ms"
            )
            if wall_elapsed > 0:
                print(
                    f"- overlap factor (sum/wall): {total_ms / (wall_elapsed * 1e3):.2f}x (closer to 1 => less overlap)"
                )
        except Exception:
            pass

    # Vectorized AUC calculation
    x = np.array(noises, dtype=float)
    for state in metrics:
        y = np.array([accuracy_dict[state][noise] for noise in noises], dtype=float)
        auc = integrate.trapezoid(y, x=x)
        accuracy_dict[state]["AUC"] = auc
        accuracy_dict["AUC"][state] = auc

    # Calculate minAUC
    accuracy_dict["AUC"]["minAUC"] = min(
        accuracy_dict["AUC"]["majority0"],
        accuracy_dict["AUC"]["majority1"],
        accuracy_dict["AUC"]["P0.5"],
    )
    gc.collect()
    return accuracy_dict


def fitness_split_auc_processpool(
    rule: str,
    n_walkers: int = 10000,  # changing n_walkers to 10000
    time_step_factor: int = 5,
    device_id=0,
    initial_config_type="normal",
    noises=None,
    lattice_size=149,
    k=7,
    parallel: bool = True,
    max_workers: int | None = None,
    debug: bool = False,
):
    """
    Function that computes the fitness of a rule

    Parameters
    ----------
    rule: The rule to evaluate
    noise: The noise level
    n_walkers: The number of simultaneous calculations
    time_step_factor: timesteps = lattice size * time_step_factor + 1

    Returns
    -------
    float: The fitness of the rule
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

    # Initialize accuracy dictionary with a factory function
    metrics = [
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
    accuracy_dict = {metric: {} for metric in metrics}
    accuracy_dict["AUC"] = {}

    worker_args = [
        (
            rule,
            noise,
            n_walkers,
            time_step_factor,
            device_id,
            initial_config_type,
            lattice_size,
            k,
            debug,
        )
        for noise in noises
    ]

    # Execute either in parallel (ProcessPool) or sequentially as fallback
    wall_start = time.perf_counter()
    per_noise_ms: dict[float, float] = {}
    try:
        if parallel and len(worker_args) > 1:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers, mp_context=ctx
            ) as executor:
                for noise, fitnesses, t_ms in executor.map(
                    _fitness_auc_processpool_worker, worker_args
                ):
                    for state, value in fitnesses.items():
                        accuracy_dict[state][noise] = value
                    per_noise_ms[noise] = t_ms
        else:
            # Sequential path
            for args in worker_args:
                noise, fitnesses, t_ms = _fitness_auc_processpool_worker(args)
                for state, value in fitnesses.items():
                    accuracy_dict[state][noise] = value
                per_noise_ms[noise] = t_ms

    except cp.cuda.runtime.CUDARuntimeError as e:
        # Fallback to sequential on GPU runtime errors
        print(f"GPU parallel error; falling back to sequential. Detail: {e}")
        for args in worker_args:
            noise, fitnesses, t_ms = _fitness_auc_processpool_worker(args)
            for state, value in fitnesses.items():
                accuracy_dict[state][noise] = value
            per_noise_ms[noise] = t_ms
    wall_elapsed = time.perf_counter() - wall_start

    if debug:
        try:
            total_ms = sum(per_noise_ms.values())
            print("\n[fitness_split_auc] Debug timings:")
            print(f"- noises: {list(noises)}")
            print(
                f"- per-noise GPU times (ms): { {round(k, 3): round(v, 2) for k, v in per_noise_ms.items()} }"
            )
            print(
                f"- sum per-noise: {total_ms:.2f} ms; wall: {wall_elapsed * 1e3:.2f} ms"
            )
            if wall_elapsed > 0:
                print(
                    f"- overlap factor (sum/wall): {total_ms / (wall_elapsed * 1e3):.2f}x (closer to 1 => less overlap)"
                )
        except Exception:
            pass

    # Vectorized AUC calculation
    x = np.array(noises, dtype=float)
    for state in metrics:
        y = np.array([accuracy_dict[state][noise] for noise in noises], dtype=float)
        auc = integrate.trapezoid(y, x=x)
        accuracy_dict[state]["AUC"] = auc
        accuracy_dict["AUC"][state] = auc

    # Calculate minAUC
    accuracy_dict["AUC"]["minAUC"] = min(
        accuracy_dict["AUC"]["majority0"],
        accuracy_dict["AUC"]["majority1"],
        accuracy_dict["AUC"]["P0.5"],
    )
    gc.collect()
    return accuracy_dict


def _fitness_auc_processpool_worker(args: tuple) -> tuple[float, dict, float]:
    (
        rule,
        noise,
        n_walkers,
        time_step_factor,
        device_id,
        initial_config_type,
        lattice_size,
        k,
        debug,
    ) = args
    noise = float(noise)
    with cp.cuda.Device(device_id):
        stream = cp.cuda.Stream(non_blocking=True)
        cp.random.seed(int((noise * 1e6) % (2**32 - 1)))
        with stream:
            if debug:
                try:
                    from cupy.cuda import nvtx  # type: ignore

                    _push = getattr(nvtx, "RangePush", None)
                    if callable(_push):
                        _push(f"noise={noise}")
                except Exception:
                    pass
            start_evt = cp.cuda.Event()
            end_evt = cp.cuda.Event()
            start_evt.record(stream)
            res = fitness_split(
                rule=rule,
                noise=noise,
                n_walkers=n_walkers,
                time_step_factor=time_step_factor,
                initial_config_type=initial_config_type,
                lattice_size=lattice_size,
                k=k,
            )
            end_evt.record(stream)
            if debug:
                try:
                    from cupy.cuda import nvtx  # type: ignore

                    _pop = getattr(nvtx, "RangePop", None)
                    if callable(_pop):
                        _pop()
                except Exception:
                    pass
        stream.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_evt, end_evt)
    return noise, res, float(elapsed_ms)


# %%
if __name__ == "__main__":
    # noise = 0.01
    # lattice_size = 149
    # k = 7
    # rule = "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011111"
    # network = ca_network_from_output_list(rule, k=k, lattice=lattice_size)
    # print(f"Network:{network}")
    # outs, ins = cw.conversions.cana2cupy_probabilisticLUT(network, prob=noise)
    # print(f"Outs.shape: {outs.shape}, ins.shape: {ins.shape}")
    # print(f"Outs: {outs}")
    # print(f"Ins: {ins}")
    # model = cw.Model(
    #     lookup_tables=outs,
    #     node_regulators=ins,
    #     n_time_steps=lattice_size * 5 + 1,
    #     n_walkers=1000,
    #     probabilistic_lut=True,
    # )
    # print("\n=========Sequential vs Parallel probability sweep timing=============\n")
    # try:
    #     # Use a valid LUT for k=3 (length 8)
    #     # sweep_rule = "01011010"
    #     sweep_rule = "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011111"  # gkl

    #     noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    #     n_walkers = 1000
    #     # First run sequentially
    #     t0 = time.perf_counter()
    #     res_seq = fitness_split_auc(
    #         rule=sweep_rule,
    #         n_walkers=n_walkers,
    #         time_step_factor=5,
    #         device_id=0,
    #         initial_config_type="normal",
    #         noises=noises,
    #         lattice_size=149,
    #         k=7,
    #         parallel=False,
    #         debug=True,
    #     )
    #     t1 = time.perf_counter()
    #     seq_time = t1 - t0
    #     print(f"Sequential time: {seq_time:.3f} s")

    #     t0 = time.perf_counter()
    #     res_par = fitness_split_auc(
    #         rule=sweep_rule,
    #         n_walkers=n_walkers,
    #         time_step_factor=5,
    #         device_id=0,
    #         initial_config_type="normal",
    #         noises=noises,
    #         lattice_size=149,
    #         k=7,
    #         parallel=True,
    #         max_workers=10,
    #         debug=True,
    #     )
    #     t1 = time.perf_counter()
    #     par_time = t1 - t0
    #     print(f"Parallel time:   {par_time:.3f} s")
    #     if par_time > 0:
    #         print(f"Speedup: {seq_time / par_time:.2f}x")
    #     # Show one metric’s AUC to confirm both paths produced outputs
    #     try:
    #         print(
    #             "AUC majority1 (seq, par):",
    #             round(res_seq["AUC"]["majority1"], 4),
    #             round(res_par["AUC"]["majority1"], 4),
    #         )
    #     except Exception:
    #         pass
    # except Exception as e:
    #     print(f"Timing sweep failed: {e}")

    # ----------------------------------------------------------------------
    # # Parameter sweep: max_workers from 1 to 10, rank speeds
    # print("\n========= max_workers sweep (1..10) and ranking =========\n")
    # sweep_stats = pd.DataFrame(columns=["n_walkers", "max_workers", "time_s"])
    # try:
    #     sweep_rule = "00000000010111110000000001011111000000000101111100000000010111110000000001011111111111110101111100000000010111111111111101011111"
    #     noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    #     for fitness_function in [fitness_split_auc_processpool, fitness_split_auc]:
    #         print(f"\n--- Testing function: {fitness_function.__name__} ---")
    #         for n_walkers in [
    #             100000,
    #             10000,
    #             5000,
    #             1000,
    #             100,
    #         ]:
    #             print(f"\n--- n_walkers={n_walkers} ---")
    #             times: dict[int, float] = {}
    #             for mw in range(1, 11):
    #                 t0 = time.perf_counter()
    #                 _ = fitness_function(
    #                     rule=sweep_rule,
    #                     n_walkers=n_walkers,
    #                     time_step_factor=5,
    #                     device_id=0,
    #                     initial_config_type="normal",
    #                     noises=noises,
    #                     lattice_size=149,
    #                     k=7,
    #                     parallel=True,
    #                     max_workers=mw,
    #                     debug=False,
    #                 )
    #                 t1 = time.perf_counter()
    #                 times[mw] = t1 - t0
    #                 print(f"max_workers={mw}: {times[mw]:.3f} s")
    #                 sweep_stats = pd.concat(
    #                     [
    #                         sweep_stats,
    #                         pd.DataFrame({
    #                             "fitness_function": [fitness_function.__name__],
    #                             "n_walkers": [n_walkers],
    #                             "max_workers": [mw],
    #                             "time_s": [times[mw]],
    #                         }),
    #                     ],
    #                     ignore_index=True,
    #                 )

    #             ranking = sorted(times.items(), key=lambda kv: kv[1])
    #             print(f"\nFor n_walkers:{n_walkers}, Ranking (fastest -> slowest):")
    #             for mw, sec in ranking:
    #                 print(f"  {mw}: {sec:.3f} s")
    #             best_mw, best_time = ranking[0]
    #             print(
    #                 f"\nBest max_workers={best_mw} (wall={best_time:.3f} s) for n_walkers={n_walkers}"
    #             )
    #         sweep_stats.to_csv("artifacts/sweep_max_workers_stats.csv", index=False)
    # except Exception as e:
    #     print(f"max_workers sweep failed: {e}")

    # load sweep stats and plot using seaborn
    print("\n========= Plotting max_workers sweep stats =========\n")
    try:
        import seaborn as sns
        from matplotlib import colors as mcolors
        import matplotlib.pyplot as plt

        sweep_stats = pd.read_csv("artifacts/sweep_max_workers_stats.csv")
        # print(sweep_stats.head())
        sns.set_palette("tab10")
        plt.figure(figsize=(10, 6))
        sweep_stats["max_workers"] = pd.to_numeric(sweep_stats["max_workers"], errors="coerce").fillna(0).astype(int)
        sweep_stats["time_per_rule"] = np.where(
            sweep_stats["max_workers"] > 0,
            sweep_stats["time_s"] / sweep_stats["max_workers"],
            np.nan,
        )
        # Treat n_walkers as a categorical variable so seaborn uses discrete hues
        sweep_stats["n_walkers"] = sweep_stats["n_walkers"].astype(str)
        # Order categories numerically for a sensible legend order
        levels = sorted(sweep_stats["n_walkers"].unique(), key=lambda v: int(v))
        sweep_stats["n_walkers"] = pd.Categorical(sweep_stats["n_walkers"], categories=levels, ordered=True)
        
        sns.lineplot(
            data=sweep_stats,
            x="max_workers",
            y="time_per_rule",
            hue="n_walkers",
            style="fitness_function",
            markers=True,
            dashes=True,
        )
        plt.title("Max Workers vs Time Taken")
        plt.xlabel("Max Workers")
        plt.ylabel("Time (s)")
        plt.grid(True)
        plt.savefig("artifacts/sweep_max_workers_plot.png")
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")
