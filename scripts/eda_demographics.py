import os
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Change working directory to project root (two levels up from scripts/)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    figures_dir = Path("notebooks/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pl.read_csv("data/kaggle_raw/training_set/demographics.csv")
    df_pd = df.to_pandas()

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}\n")

    # --- Console stats ---

    ci_counts = df["Cognitive_Impairment"].value_counts().sort("Cognitive_Impairment")
    print("Cognitive_Impairment Distribution:")
    for row in ci_counts.iter_rows(named=True):
        pct = row["count"] / df.height * 100
        print(f"  {row['Cognitive_Impairment']}: {row['count']} ({pct:.2f}%)")

    df_true = df.filter(pl.col("Cognitive_Impairment") == True)
    if not df_true.is_empty():
        tte = df_true["Time_to_Event"]
        print("\nTime_to_Event Stats (True cases):")
        print(f"  Mean: {tte.mean():.2f}  Std: {tte.std():.2f}")
        print(f"  Min:  {tte.min():.2f}  Max: {tte.max():.2f}")

    print("\nSite Distribution:")
    for row in df["SiteID"].value_counts().sort("SiteID").iter_rows(named=True):
        print(f"  {row['SiteID']}: {row['count']}")

    age = df["Age"]
    print("\nAge Stats:")
    print(f"  Mean: {age.mean():.2f}  Std: {age.std():.2f}")
    print(f"  Min:  {age.min():.2f}  Max: {age.max():.2f}")

    print("\nSex Distribution:")
    for row in df["Sex"].value_counts().sort("Sex").iter_rows(named=True):
        pct = row["count"] / df.height * 100
        print(f"  {row['Sex']}: {row['count']} ({pct:.2f}%)")

    # --- Figure 1: Overall demographics overview ---

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    ci_counts_pd = df_pd["Cognitive_Impairment"].value_counts().sort_index()
    sns.barplot(x=ci_counts_pd.index.astype(str), y=ci_counts_pd.values, ax=axes[0])
    axes[0].set_title("Cognitive_Impairment Distribution")
    axes[0].set_xlabel("Impairment")
    axes[0].set_ylabel("Count")

    df_true_pd = df_pd[df_pd["Cognitive_Impairment"] == True]
    if not df_true_pd.empty:
        sns.histplot(df_true_pd["Time_to_Event"], kde=True, ax=axes[1])
        axes[1].set_title("Time_to_Event Distribution (True Cases)")
    else:
        axes[1].text(0.5, 0.5, "No True Cases", ha="center", va="center")
    axes[1].set_xlabel("Time to Event (days)")

    site_counts_pd = df_pd["SiteID"].value_counts().sort_index()
    sns.barplot(x=site_counts_pd.index, y=site_counts_pd.values, ax=axes[2])
    axes[2].set_title("Site Distribution")
    axes[2].set_xlabel("Site")
    axes[2].set_ylabel("Count")

    sns.histplot(data=df_pd, x="Age", hue="Cognitive_Impairment",
                 multiple="stack", ax=axes[3])
    axes[3].set_title("Age by CI Status")
    axes[3].set_xlabel("Age")
    axes[3].set_ylabel("Count")

    sex_counts_pd = df_pd["Sex"].value_counts().sort_index()
    sns.barplot(x=sex_counts_pd.index.astype(str), y=sex_counts_pd.values, ax=axes[4])
    axes[4].set_title("Sex Distribution")
    axes[4].set_xlabel("Sex")
    axes[4].set_ylabel("Count")

    axes[5].axis("off")

    plt.tight_layout()
    out = figures_dir / "demographics_overview.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # --- Figure 2: Age and TTE stratified by sex ---

    sexes = sorted(df_pd["Sex"].dropna().unique())
    fig, axes = plt.subplots(len(sexes), 2, figsize=(12, 4 * len(sexes)))

    for i, sex in enumerate(sexes):
        subset = df_pd[df_pd["Sex"] == sex]

        sns.histplot(data=subset, x="Age", hue="Cognitive_Impairment",
                     multiple="stack", ax=axes[i, 0])
        axes[i, 0].set_title(f"{sex} — Age by CI Status")
        axes[i, 0].set_xlabel("Age")

        tte_subset = subset[subset["Cognitive_Impairment"] == True]["Time_to_Event"].dropna()
        if not tte_subset.empty:
            sns.histplot(tte_subset, kde=True, ax=axes[i, 1])
        else:
            axes[i, 1].text(0.5, 0.5, "No CI cases", ha="center", va="center")
        axes[i, 1].set_title(f"{sex} — Time to Event (CI=True)")
        axes[i, 1].set_xlabel("Days")

    plt.tight_layout()
    out = figures_dir / "demographics_by_sex.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()