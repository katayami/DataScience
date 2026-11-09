"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def load_brain_data(csv_path: str) -> pd.DataFrame:
    """Загружает brainsize.txt: разделитель tab, пропуски обозначены NA."""
    df = pd.read_csv(csv_path, sep="\t", na_values=["NA"])
    df["Gender"] = df["Gender"].str.strip()
    return df


def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    corr = group[FEATURES + [TARGET]].corr(method="pearson")[TARGET]
    return {feature: float(corr[feature]) for feature in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = load_brain_data(data.csv_path)
    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_corr = _mri_correlations(men)
    women_corr = _mri_correlations(women)

    candidates = [
        (abs(value), feature)
        for corr in (women_corr, men_corr)
        for feature, value in corr.items()
        if pd.notna(value)
    ]
    strongest_mri_feature = max(candidates)[1]

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_mri_feature,
    )
