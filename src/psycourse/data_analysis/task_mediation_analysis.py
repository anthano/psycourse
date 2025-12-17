from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.mediation_analysis import mediation_analysis

# Define all PRS keys and their task kwargs.
PRS_TASKS = {
    prs: {
        "prs": prs,
        "produces": BLD_RESULTS / "mediation_analysis" / f"mediation_results_{prs}.pkl",
    }
    for prs in [
        "SCZ_PRS",
        "BD_PRS",
        "MDD_PRS",
        "Lipid_SCZ_PRS",
        "Lipid_BD_PRS",
        "Lipid_MDD_PRS",
    ]
}

# Define a single task template and use kwargs to customize behavior.
for prs_id, kwargs in PRS_TASKS.items():

    @task(id=prs_id, kwargs=kwargs)
    def task_mediation_analysis(
        prs: str,
        multimodal_df_path: Path = BLD_DATA / "multimodal_complete_df.pkl",
        produces: Annotated[Path, Product] = None,
    ) -> None:
        df = pd.read_pickle(multimodal_df_path)
        med_res = mediation_analysis(df, prs)
        med_res.to_pickle(produces)
