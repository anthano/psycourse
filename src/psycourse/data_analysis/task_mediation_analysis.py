from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task

from psycourse.config import BLD_DATA, BLD_RESULTS
from psycourse.data_analysis.mediation_analysis import mediation_analysis

ALL_PRS = ["SCZ_PRS", "BD_PRS", "MDD_PRS"]
LIPID_PRS = ["Lipid_SCZ_PRS", "Lipid_BD_PRS", "Lipid_MDD_PRS"]

for prs in ALL_PRS:

    @task(id=prs)
    def task_mediation_analysis(
        multimodal_df_path: Path = BLD_DATA / "multimodal_complete_df.pkl",
        produces: Annotated[Path, Product] = BLD_RESULTS
        / "mediation_analysis"
        / f"mediation_results_{prs}.pkl",
    ) -> None:
        df = pd.read_pickle(multimodal_df_path)
        med_res = mediation_analysis(df, prs)  # noqa: B023
        med_res.to_pickle(produces)


for lipid_prs in LIPID_PRS:

    @task(id=lipid_prs)
    def task_lipid_mediation_analysis(
        multimodal_df_path: Path = BLD_DATA / "multimodal_complete_df.pkl",
        produces: Annotated[Path, Product] = BLD_RESULTS
        / "mediation_analysis"
        / f"mediation_results_{lipid_prs}.pkl",
    ) -> None:
        df = pd.read_pickle(multimodal_df_path)
        med_res = mediation_analysis(df, lipid_prs)  # noqa: B023
        med_res.to_pickle(produces)
