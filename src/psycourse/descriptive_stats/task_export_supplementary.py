import pandas as pd
from pytask import task

from psycourse.config import BLD_RESULTS, WRITING


@task(
    kwargs={
        "depends_on": {
            "lipid": BLD_RESULTS / "tables" / "lipid_data_table.pkl",
            "demographics": BLD_RESULTS / "descriptive_stats" / "lipid_data_table.pkl",
            # ... add other tables here
        },
        "produces": WRITING / "supplementary_tables.xlsx",
    }
)
def task_export_all_supplementary_tables(depends_on, produces):
    with pd.ExcelWriter(produces, engine="openpyxl") as writer:
        for name, path in depends_on.items():
            df = pd.read_pickle(path)
            # Sheet names limited to 31 chars in Excel
            sheet_name = f"Table_{name}"[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=True)
