import json
from itertools import product
from pathlib import Path

import pandas as pd
import pytask

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.hurdle_wrappers import run_single_combo_repeat

# TODO: UPDATE ALL THIS WITH BETTER VARIABLES, CLEANER PATHS ETC

DATA = BLD_DATA / "multimodal_complete_df.pkl"
HURDLE_OUTPUT_DIRECTORY = BLD_RESULTS / "multivariate" / "hurdle_runs"
HURDLE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

CUTOFFS = [0.05, 0.10, 0.15, 0.25]
INNERS = [5, 7, 10]
OUTERS = [5, 7, 10]
N_REPEATS = 10

# ---------------- per-repeat tasks (new loops syntax) ----------------
for cutoff, inner, outer, repeat in product(CUTOFFS, INNERS, OUTERS, range(N_REPEATS)):
    combo_id = f"{cutoff}_{inner}_{outer}"
    run_dir = HURDLE_OUTPUT_DIRECTORY / combo_id
    out_json = run_dir / f"repeat_{repeat}.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    @pytask.task
    def task_hurdle_repeat(
        depends_on=DATA,
        produces=out_json,
        _cutoff=cutoff,
        _inner=inner,
        _outer=outer,
        _repeat=repeat,
    ):
        df = pd.read_pickle(depends_on)
        metrics = run_single_combo_repeat(
            df,
            _cutoff,
            _inner,
            _outer,
            _repeat,
            base_seed=42,
            clf_n_jobs=1,
            reg_n_jobs=1,
        )
        Path(produces).parent.mkdir(parents=True, exist_ok=True)
        Path(produces).write_text(json.dumps(metrics, indent=2))
        assert Path(produces).is_file()


# ---------------- aggregation per combo ----------------
for cutoff, inner, outer in product(CUTOFFS, INNERS, OUTERS):
    combo_id = f"{cutoff}_{inner}_{outer}"
    run_dir = HURDLE_OUTPUT_DIRECTORY / combo_id
    repeat_files = [run_dir / f"repeat_{r}.json" for r in range(N_REPEATS)]
    summary_json = run_dir / "summary.json"
    per_repeat_csv = run_dir / "per_repeat.csv"

    @pytask.task
    def task_hurdle_aggregate(
        depends_on=repeat_files,
        produces=(summary_json, per_repeat_csv),
        _cutoff=cutoff,
        _inner=inner,
        _outer=outer,
    ):
        rows = [json.loads(Path(p).read_text()) for p in depends_on if Path(p).exists()]
        per_repeat = pd.DataFrame(rows)
        per_repeat.to_csv(produces[1], index=False)

        def mean_std(col):
            return float(per_repeat[col].mean()), float(per_repeat[col].std(ddof=1))

        auc_mean, auc_std = mean_std("test_roc_auc")
        accuracy_mean, accuracy_std = mean_std("test_accuracy")
        precision_mean, precision_std = mean_std("test_precision")
        recall_mean, recall_std = mean_std("test_recall")
        r2_mean, r2_std = mean_std("test_r2")
        mse_mean, mse_std = mean_std("test_mse")
        permutation_pvalue_mean, permutation_std = mean_std("permutation_pvalue")
        # permutation p value mean -> should that not be multiple testing corrected?
        summary = {
            "combo_id": f"{_cutoff}_{_inner}_{_outer}",
            "cutoff_quantile": float(_cutoff),
            "n_inner_cv": int(_inner),
            "n_outer_cv": int(_outer),
            "n_repeats": int(len(per_repeat)),
            "AUC_test_mean": auc_mean,
            "AUC_test_std": auc_std,
            "ACC_test_mean": accuracy_mean,
            "ACC_test_std": accuracy_std,
            "PREC_test_mean": precision_mean,
            "PREC_test_std": precision_std,
            "REC_test_mean": recall_mean,
            "REC_test_std": recall_std,
            "R2_test_mean": r2_mean,
            "R2_test_std": r2_std,
            "MSE_test_mean": mse_mean,
            "MSE_test_std": mse_std,
            "perm_p_mean": permutation_pvalue_mean,
            "perm_p_std": permutation_std,
        }
        Path(produces[0]).write_text(json.dumps(summary, indent=2))
        assert Path(produces[0]).is_file()
        assert Path(produces[1]).is_file()


# ---------------- final collection across all combos ----------------
@pytask.task
def task_hurdle_collect(produces=HURDLE_OUTPUT_DIRECTORY / "summary_all.csv"):
    combo_dirs = [path for path in HURDLE_OUTPUT_DIRECTORY.iterdir() if path.is_dir()]
    summaries = []
    for directory in combo_dirs:
        summary_json = directory / "summary.json"
        if summary_json.exists():
            summaries.append(json.loads(summary_json.read_text()))
    pd.DataFrame(summaries).to_csv(produces, index=False)
    assert Path(produces).is_file()
