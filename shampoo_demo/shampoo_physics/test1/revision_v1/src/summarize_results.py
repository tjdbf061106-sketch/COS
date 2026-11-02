"""Descriptive summaries and figures, without model or threshold selection."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from research_core import ROOT, METRICS, read_json, write_json

RESULTS = ROOT / "results"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 180})


def run():
    rep = pd.read_csv(RESULTS / "representation_repeat_metrics.csv")
    rows = []
    for (features, model, target), group in rep.groupby(["features", "model", "target"]):
        row = {"features": features, "model": model, "target": target, "n_repeats": len(group)}
        for metric in METRICS if target == "stability" else ["MAE_log", "RMSE_log", "R2_log"]:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_repeat_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    rep_summary = pd.DataFrame(rows)
    rep_summary.to_csv(RESULTS / "representation_summary.csv", index=False, encoding="utf-8-sig")
    novel = pd.read_csv(RESULTS / "onboarding_metrics.csv", dtype={"budget": str})
    keys = ["held_ingredient", "features", "budget", "target", "cohort"]
    selected = METRICS + ["MAE_log", "RMSE_log", "R2_log"]
    per_material = novel.groupby(keys)[selected].mean().reset_index()
    per_material.to_csv(RESULTS / "onboarding_per_material.csv", index=False, encoding="utf-8-sig")
    macro = per_material.groupby(["features", "budget", "target", "cohort"])[selected].mean().reset_index()
    macro.to_csv(RESULTS / "onboarding_macro_summary.csv", index=False, encoding="utf-8-sig")
    complete = per_material[per_material.budget == "all"]
    zero = per_material[per_material.budget == "0"]
    pairs = complete.merge(zero, on=["held_ingredient", "features", "target", "cohort"], suffixes=("_all", "_zero"), validate="one_to_one")
    for metric in selected: pairs[metric + "_delta_all_minus_zero"] = pairs[metric + "_all"] - pairs[metric + "_zero"]
    pairs.to_csv(RESULTS / "onboarding_paired_changes.csv", index=False, encoding="utf-8-sig")
    allocation = pd.read_csv(RESULTS / "onboarding_allocation.csv", dtype={"budget": str})
    allocation["zero_shot"] = allocation.budget.eq("0")
    allocation["no_new_stable_observation"] = allocation.adaptation_stable.eq(0) & ~allocation.zero_shot
    allocation["no_new_unstable_observation"] = allocation.adaptation_stable.eq(allocation.adaptation_recipes) & ~allocation.zero_shot
    allocation["manufacturing_batch_independence"] = "unresolved"
    allocation.to_csv(RESULTS / "onboarding_readiness_flags.csv", index=False, encoding="utf-8-sig")
    selective = pd.read_csv(RESULTS / "selective_prediction_metrics.csv")
    selective_cols = ["coverage", "stable_coverage", "unstable_coverage", *METRICS, "selective_error"]
    selective_summary = selective.groupby(["features", "model", "confidence_margin"])[selective_cols].mean().reset_index()
    selective_summary.to_csv(RESULTS / "selective_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=False, constrained_layout=True)
    labels = [("stock18", "linear"), ("physics_core61", "linear"), ("stock18", "extra_trees"), ("physics_core61", "extra_trees")]
    display = ["18 / Linear", "61 / Linear", "18 / Trees", "61 / Trees"]
    c = rep_summary[rep_summary.target == "stability"].set_index(["features", "model"])
    for ax, metric in zip(axes, METRICS):
        means = [c.loc[key, metric + "_mean"] for key in labels]
        sd = [c.loc[key, metric + "_repeat_sd"] for key in labels]
        ax.bar(range(4), means, yerr=sd, capsize=3, color=["#9ab6cc", "#176b8e", "#a8c9b8", "#218571"])
        ax.set_xticks(range(4), display, rotation=45, ha="right")
        ax.set_title(metric); ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Representation comparison | positive=stable, threshold=0.5\nError bars: split-repeat SD (3 repeats), not confidence intervals", fontsize=12)
    fig.savefig(RESULTS / "representation_four_metrics.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    budgets = ["0", "4", "12", "all"]
    for feature, color in [("stock18", "#176b8e"), ("physics_core61", "#b65c37")]:
        points = macro[(macro.features == feature) & (macro.target == "stability") & (macro.cohort == "new_final_test")].set_index("budget")
        for ax, metric in zip(axes, METRICS):
            ax.plot(range(4), points.loc[budgets, metric], "o-", label=feature, color=color)
            ax.set_xticks(range(4), ["0", "4", "12", "All pool"]); ax.set_xlabel("Added combination groups")
            ax.set_title(metric); ax.set_ylim(0, 1); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Simulated new-material onboarding | linear models\nDescriptive equal-material means; final test fixed; not external validation", fontsize=12)
    fig.savefig(RESULTS / "onboarding_learning_curves.png"); plt.close(fig)

    s = selective_summary[(selective_summary.features == "stock18") & (selective_summary.model == "extra_trees")].sort_values("confidence_margin")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(s.coverage, s.selective_error, "o-", color="#176b8e")
    axes[0].set(xlabel="Fraction receiving a prediction", ylabel="Error among accepted predictions", title="Error-coverage trade-off")
    axes[1].plot(s.confidence_margin, s.stable_coverage, "o-", label="Stable")
    axes[1].plot(s.confidence_margin, s.unstable_coverage, "s-", label="Unstable")
    axes[1].set(xlabel="Fixed confidence margin", ylabel="Fraction accepted within class", title="Rejection is class-dependent")
    axes[1].legend()
    for metric in METRICS: axes[2].plot(s.confidence_margin, s[metric], "o-", label=metric)
    axes[2].set(xlabel="Fixed confidence margin", ylabel="Accepted-subset metric", title="Report all four metrics")
    axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("Exploratory confidence filtering | stock18 / Extra Trees\nNo operating threshold selected or deployed", fontsize=12)
    fig.savefig(RESULTS / "selective_coverage.png"); plt.close(fig)

    panel = pairs[(pairs.features == "stock18") & (pairs.target == "stability") & (pairs.cohort == "new_final_test")].sort_values("held_ingredient")
    values = panel[[m + "_delta_all_minus_zero" for m in METRICS]].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 9), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax.set_yticks(range(len(panel)), panel.held_ingredient.tolist())
    ax.set_xticks(range(4), METRICS)
    for i in range(len(panel)):
        for j in range(4): ax.text(j, i, f"{values[i,j]:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(values[i,j]) > .6 else "black")
    fig.colorbar(im, ax=ax, label="All adaptation pool minus zero new observations")
    ax.set_title("Material-specific changes | stock18 / linear\nHigher is better for these four metrics; effects vary by material")
    fig.savefig(RESULTS / "material_change_map.png"); plt.close(fig)

    counts = {}
    for feature in ["stock18", "physics_core61"]:
        g = pairs[(pairs.features == feature) & (pairs.target == "stability") & (pairs.cohort == "new_final_test")]
        counts[feature] = {"materials_with_MCC_increase": int((g.MCC_delta_all_minus_zero > 0).sum()),
                           "materials_with_both_class_recalls_increased": int(((g.Sensitivity_delta_all_minus_zero > 0) & (g.Specificity_delta_all_minus_zero > 0)).sum())}
    write_json(RESULTS / "descriptive_findings.json", {"all_materials_reported": 18, "interpretation": "No significance or independent-material sampling claim",
        "material_change_counts": counts,
        "four_group_cases_without_new_stable_observations": int(allocation[allocation.budget == "4"].no_new_stable_observation.sum()),
        "four_group_cases": int(allocation.budget.eq("4").sum()),
        "onboarding_regression_and_old_retention_reported": True,
        "fit_counts": {"representations": read_json(RESULTS / "representation_run.json")["fits"], "onboarding": read_json(RESULTS / "onboarding_run.json")["fits"]}})
    print("Descriptive tables and four standalone figures saved; no model/threshold selection performed.")


if __name__ == "__main__": run()
