from pathlib import Path

import pandas as pd

THIS_DIR = Path(".").resolve()
ROOT = THIS_DIR.parent.parent.resolve()
DATA_DIR = ROOT / "src" / "data"
LIPIDOMICS_DIR = DATA_DIR / "lipidomics"
BLD_DATA = ROOT / "bld" / "data"
BLD_DATA.mkdir(parents=True, exist_ok=True)


def clean_data(sample_description, prs, lipid_intensities, cluster_labels):
    """Cleans and merges the sample description, Prs, lipidomic and clustering data.


    Args:
    PRS_df: pd.DataFrame with the PRS data
    lipidomics_df: pd.DataFrame with the lipidomics data
    cluster_df: pd.DataFrame with the clustering data

    Returns:
    clean_data: pd.DataFrame
    """

    clean_cluster_labels = _clean_cluster_labels(cluster_labels)
    clean_prs = _clean_prs(prs)
    clean_sample_description = _clean_sample_description(sample_description)
    clean_lipid_intensities = _clean_lipid_intensities(lipid_intensities)

    clean_data = pd.merge(
        clean_lipid_intensities, clean_prs, on=["Patient_ID", "cluster_label"]
    )

    return {
        "clean_cluster_labels": clean_cluster_labels,
        "clean_prs": clean_prs,
        "clean_sample_description": clean_sample_description,
        "clean_lipid_intensities": clean_lipid_intensities,
        "clean_data": clean_data,
    }


def _clean_sample_description(sample_description):
    sample_description = sample_description.set_index("Patient_ID")

    clean_sample_description = pd.DataFrame(index=sample_description.index)

    clean_sample_description["sex"] = sample_description["sex"].astype("category")
    clean_sample_description["age"] = sample_description["age"].astype("int")
    clean_sample_description["bmi"] = sample_description["bmi"]
    clean_sample_description["clinic"] = sample_description["clinic"].astype("category")
    clean_sample_description["year"] = sample_description["year"].astype("int")
    clean_sample_description["diagnosis"] = sample_description["diagnosis"].astype(
        "category"
    )

    return clean_sample_description


def _clean_cluster_labels(cluster_labels):
    cluster_labels = cluster_labels.set_index("cases")
    cluster_labels.index.name = "Patient_ID"

    clean_cluster_labels = pd.DataFrame(index=cluster_labels.index)

    clean_cluster_labels["cluster_label"] = cluster_labels["cluster_label"].astype(
        "category"
    )

    return clean_cluster_labels


def _clean_prs(prs):
    prs = prs.set_index("ID")
    prs.index.name = "Patient_ID"

    clean_prs = pd.DataFrame(index=prs.index)
    clean_prs["cluster_label"] = prs["group"].str.strip("subtype_").astype("category")
    for col in prs.filter(like="PRS_").columns:
        clean_prs[col] = prs[col].astype("float")

    return clean_prs


def _clean_lipid_intensities(lipid_intensities):
    clean_lipid_intensities = lipid_intensities.copy()
    clean_lipid_intensities = lipid_intensities.set_index("ID")
    clean_lipid_intensities.index.name = "Patient_ID"
    clean_lipid_intensities["group"] = (
        clean_lipid_intensities["group"].str.strip("subtype_").astype("category")
    )
    clean_lipid_intensities = clean_lipid_intensities.rename(
        columns={"group": "cluster_label"}
    )

    return clean_lipid_intensities


if __name__ == "__main__":
    sample_description = pd.read_csv(
        LIPIDOMICS_DIR / "sample_description.csv", delimiter=";"
    )
    lipid_intensities = pd.read_csv(DATA_DIR / "lipidomics.csv", delimiter=",")
    cluster_labels = pd.read_csv(DATA_DIR / "ClusterLabels.csv", delimiter=",")
    prs = pd.read_csv(DATA_DIR / "PRS.csv", delimiter=",")

    cleaned_df = clean_data(sample_description, prs, lipid_intensities, cluster_labels)

    for key, value in cleaned_df.items():
        value.to_csv(BLD_DATA / f"{key}.csv", sep=";", index=True)
        value.to_pickle(BLD_DATA / f"{key}.pkl")
