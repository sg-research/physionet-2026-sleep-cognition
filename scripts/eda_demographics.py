import os
import sys
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # 1. Change working directory to the project root (two levels up from scripts/)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # Ensure output directory exists
    Path("notebooks/figures").mkdir(parents=True, exist_ok=True)

    # 2. Load data with polars
    df = pl.read_csv("data/kaggle_raw/training_set/demographics.csv")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}\n")

    # Cognitive_Impairment label distribution (True/False counts and %)
    ci_counts = df["Cognitive_Impairment"].value_counts().sort("Cognitive_Impairment")
    print("Cognitive_Impairment Distribution:")
    for row in ci_counts.iter_rows(named=True):
        val = row["Cognitive_Impairment"]
        count = row["count"]
        pct = count / df.height * 100
        print(f"  {val}: {count} ({pct:.2f}%)")

    # Time_to_Event distribution stats for True cases
    df_true = df.filter(pl.col("Cognitive_Impairment") == True)
    if not df_true.is_empty():
        tte = df_true["Time_to_Event"]
        print("\nTime_to_Event Stats (True cases):")
        print(f"  Mean: {tte.mean():.2f}")
        print(f"  Std: {tte.std():.2f}")
        print(f"  Min: {tte.min():.2f}")
        print(f"  Max: {tte.max():.2f}")
    else:
        print("\nTime_to_Event Stats (True cases): No True cases found.")

    # Site distribution (S0001, I0002, I0006 counts)
    site_counts = df["Site"].value_counts().sort("Site")
    print("\nSite Distribution:")
    for row in site_counts.iter_rows(named=True):
        print(f"  {row['Site']}: {row['count']}")

    # Age mean/std/min/max
    age = df["Age"]
    print("\nAge Stats:")
    print(f"  Mean: {age.mean():.2f}")
    print(f"  Std: {age.std():.2f}")
    print(f"  Min: {age.min():.2f}")
    print(f"  Max: {age.max():.2f}")

    # Sex distribution
    sex_counts = df["Sex"].value_counts().sort("Sex")
    print("\nSex Distribution:")
    for row in sex_counts.iter_rows(named=True):
        val = row["Sex"]
        count = row["count"]
        pct = count / df.height * 100
        print(f"  {val}: {count} ({pct:.2f}%)")

    # 3. Save figure grid
    # Convert to pandas for seaborn compatibility
    df_pd = df.to_pandas()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Cognitive_Impairment bar chart
    ci_counts_pd = df_pd["Cognitive_Impairment"].value_counts().sort_index()
    sns.barplot(x=ci_counts_pd.index.astype(str), y=ci_counts_pd.values, ax=axes[0])
    axes[0].set_title("Cognitive_Impairment Distribution")
    axes[0].set_xlabel("Impairment")
    axes[0].set_ylabel("Count")

    # Time_to_Event histogram (True cases only)
    df_true_pd = df_pd[df_pd["Cognitive_Impairment"] == True]
    if not df_true_pd.empty:
        sns.histplot(df_true_pd["Time_to_Event"], kde=True, ax=axes[1])
        axes[1].set_title("Time_to_Event Distribution (True Cases)")
    else:
        axes[1].text(0.5, 0.5, "No True Cases", ha="center", va="center")
    axes[1].set_xlabel("Time to Event")

    # Site distribution bar chart
    site_counts_pd = df_pd["Site"].value_counts().sort_index()
    sns.barplot(x=site_counts_pd.index, y=site_counts_pd.values, ax=axes[2])
    axes[2].set_title("Site Distribution")
    axes[2].set_xlabel("Site")
    axes[2].set_ylabel("Count")

    # Age histogram by CI status
    df_age_ci = df_pd[["Age", "Cognitive_Impairment"]].rename(columns={"Cognitive_Impairment": "CI"})
    sns.histplot(data=df_age_ci, x="Age", hue="CI", multiple="stack", ax=axes[3])
    axes[3].set_title("Age Histogram by CI Status")
    axes[3].set_xlabel("Age")
    axes[3].set_ylabel("Count")

    # Sex distribution
    sex_counts_pd = df_pd["Sex"].value_counts().sort_index()
    sns.barplot(x=sex_counts_pd.index.astype(str), y=sex_counts_pd.values, ax=axes[4])
    axes[4].set_title("Sex Distribution")
    axes[4