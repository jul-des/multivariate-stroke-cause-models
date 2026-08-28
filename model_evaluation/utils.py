import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from mrmr import mrmr_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


# ---------------------------------------------------------------------------
# Perfect separation check
# ---------------------------------------------------------------------------

def _has_perfect_separation(x, y):
    """
    Check for perfect (or quasi-perfect) separation for a single feature.

    For a binary feature: checks whether one category maps exclusively to one
    outcome class (i.e. a cell in the 2x2 contingency table is zero).

    For a continuous feature: checks whether the value ranges of the two
    outcome classes are entirely non-overlapping.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    unique_x = np.unique(x)

    # Binary feature — zero cell in contingency table
    if set(unique_x).issubset({0, 1}):
        for val in unique_x:
            mask = x == val
            if len(np.unique(y[mask])) == 1:
                return True
        return False

    # Continuous feature — non-overlapping ranges across outcome classes
    x0 = x[y == 0]
    x1 = x[y == 1]

    if len(x0) == 0 or len(x1) == 0:
        return True  # only one outcome class present

    return max(x0) < min(x1) or max(x1) < min(x0)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mrmr import mrmr_classif


# ----------------------------------------------------------------------
# Fixed color per feature category. No legend is drawn; instead the
# y-axis tick labels themselves are colored to match, so the category
# of each feature is identifiable directly from its label.
# ----------------------------------------------------------------------
CATEGORY_COLORS = {
    "Clinical":            "#4C72B0",  # blue
    "Shape-based":         "#55A868",  # green
    "Distribution-based":  "#C44E52",  # red
    "Lesion-based":        "#8172B2",  # purple
}

# "Nice" round values used to pick x-axis tick locations on the log-scale
# OR axis.
_NICE_TICK_VALUES = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
]


def _nice_floor(x, nice_values=_NICE_TICK_VALUES):
    """Largest nice value <= x (falls back to the smallest nice value if none)."""
    candidates = [v for v in nice_values if v <= x]
    return max(candidates) if candidates else min(nice_values)


def _nice_ceil(x, nice_values=_NICE_TICK_VALUES):
    """Smallest nice value >= x (falls back to the largest nice value if none)."""
    candidates = [v for v in nice_values if v >= x]
    return min(candidates) if candidates else max(nice_values)


def _assign_categories(features, clinical, shape_based, distribution_based, lesion_based):
    """
    Map each feature to exactly one category based on the provided lists.

    Returns (feature_to_category, category_order):
      - (None, None) if none of the four grouping lists were supplied
        (grouping/coloring is skipped entirely, original behavior).
      - Otherwise every feature in `features` MUST appear in exactly one
        of the non-None lists, or a ValueError is raised.
    """
    group_lists = {
        "Clinical":           clinical,
        "Shape-based":        shape_based,
        "Distribution-based": distribution_based,
        "Lesion-based":       lesion_based,
    }

    if all(v is None for v in group_lists.values()):
        return None, None

    feature_to_category = {}
    for feature in features:
        matches = [
            name for name, lst in group_lists.items()
            if lst is not None and feature in lst
        ]
        if len(matches) == 0:
            raise ValueError(
                f"Feature '{feature}' was not found in any of the provided "
                f"category lists (clinical, shape_based, distribution_based, "
                f"lesion_based). All selected features must be assigned to "
                f"exactly one category."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Feature '{feature}' appears in multiple category lists: "
                f"{matches}. Each feature must belong to exactly one category."
            )
        feature_to_category[feature] = matches[0]

    # Preserve a stable category order: clinical, shape, distribution, lesion,
    # restricted to the categories that were actually supplied.
    category_order = [name for name, lst in group_lists.items() if lst is not None]

    return feature_to_category, category_order


def _sort_grouped(df, category_order):
    """Sort by category order (if grouping is active), then by OR within each group."""
    if category_order is None:
        return df.sort_values("OR")

    df = df.copy()
    df["category"] = pd.Categorical(df["category"], categories=category_order, ordered=True)
    return df.sort_values(["category", "OR"])


def run_final_training_pipeline(
    X,
    y,
    id_col="Patient ID",
    n_features=15,
    C=0.1,
    n_bootstrap=1000,
    random_state=42,
    feature_name_map=None,
    clinical=None,
    shape_based=None,
    distribution_based=None,
    lesion_based=None,
    fontsize=11,
    figsize=None,
    max_rows=20,
):
    """
    clinical, shape_based, distribution_based, lesion_based:
        Optional lists of feature names. If any of these are supplied,
        every feature selected by mRMR must appear in exactly one of them;
        features are then grouped by category (sorted by OR within each
        group) and color-coded in both the univariate and multivariable
        forest plots, with matching colors applied to the y-axis labels.
        If none are supplied, plotting falls back to the original
        (ungrouped, single-color) behavior.
    fontsize:
        Base font size (points) used for axis labels, tick labels, the
        title, and the OR/CI text annotations in both forest plots.
    figsize:
        Fixed (width, height) in inches for both forest plots. If None
        (default), derived automatically from max_rows so row spacing
        stays identical across runs.
    max_rows:
        The maximum number of features you'd ever plot (e.g. your largest
        n_features). Row spacing is fixed based on this value rather than
        the actual number of features selected, so a run with fewer
        features (e.g. 15 out of a max of 20) simply shows blank rows at
        the bottom instead of compressed/expanded row spacing.
    """
    np.random.seed(random_state)

    X = X.drop(columns=[id_col])

    # ------------------------------------------------------------------
    # mRMR feature selection
    # ------------------------------------------------------------------
    selected_features = mrmr_classif(X=X, y=y, K=n_features)

    print("\nSelected features from mRMR:")
    print(selected_features)

    X = X[selected_features].copy()

    # ------------------------------------------------------------------
    # Resolve feature -> category mapping (raises if incomplete)
    # ------------------------------------------------------------------
    feature_to_category, category_order = _assign_categories(
        selected_features, clinical, shape_based, distribution_based, lesion_based
    )

    # ------------------------------------------------------------------
    # Detect binary vs continuous features
    # ------------------------------------------------------------------
    binary_features = []
    continuous_features = []

    for col in X.columns:
        unique_vals = set(X[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            binary_features.append(col)
        else:
            continuous_features.append(col)

    print("\nBinary features (not standardized):")
    print(binary_features)
    print("\nContinuous features (standardized):")
    print(continuous_features)

    # ------------------------------------------------------------------
    # Standardize continuous features
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_processed = X.copy()

    if continuous_features:
        X_processed[continuous_features] = scaler.fit_transform(
            X_processed[continuous_features]
        )

    n = X_processed.shape[0]

    # ------------------------------------------------------------------
    # Multivariable logistic regression  (L2, bootstrap CIs — unchanged)
    # ------------------------------------------------------------------
    model = LogisticRegression(
        C=C, penalty="l2", solver="liblinear", max_iter=1000
    )
    model.fit(X_processed, y)
    coef = model.coef_[0]

    print(f"\nRunning {n_bootstrap} bootstrap iterations for multivariable model ...")
    y_arr_multi = np.asarray(y, dtype=float)
    boot_coefs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        m = LogisticRegression(
            C=C, penalty="l2", solver="liblinear", max_iter=1000
        )
        m.fit(X_processed.iloc[idx], y_arr_multi[idx])
        boot_coefs.append(m.coef_[0])

    boot_coefs = np.array(boot_coefs)
    lower = np.percentile(boot_coefs, 2.5, axis=0)
    upper = np.percentile(boot_coefs, 97.5, axis=0)

    multi_results = pd.DataFrame({
        "feature":   selected_features,
        "OR":        np.exp(coef),
        "CI_lower":  np.exp(lower),
        "CI_upper":  np.exp(upper),
    })

    if feature_to_category is not None:
        multi_results["category"] = multi_results["feature"].map(feature_to_category)

    if feature_name_map is not None:
        multi_results["display_name"] = multi_results["feature"].map(
            lambda x: feature_name_map.get(x, x)
        )
    else:
        multi_results["display_name"] = multi_results["feature"]

    # Printing and plotting of the multivariable results is deferred until
    # after the univariate analysis below, so the multivariable plot can be
    # reordered to match the univariate feature order.

    # ------------------------------------------------------------------
    # Univariate logistic regression via statsmodels
    # ------------------------------------------------------------------
    print("\nFitting univariate logistic regression (statsmodels) …")

    uni_results = []
    separated_features = []
    y_arr = np.asarray(y, dtype=float)

    for feature in selected_features:
        x_col = X_processed[feature].values

        # MedHist HF is known to exhibit perfect separation in this cohort.
        # It's excluded from the univariate analysis explicitly (in addition
        # to the automatic check below, as a safeguard in case the automatic
        # detection ever misses it).
        if feature == "MedHist HF" or _has_perfect_separation(x_col, y_arr):
            separated_features.append(feature)
            print(f"  [SKIPPED] '{feature}' — perfect separation detected.")
            continue

        X_uni = sm.add_constant(x_col, has_constant="add")

        try:
            res = sm.Logit(y_arr, X_uni).fit(disp=False, method="newton")

            try:
                ci = res.conf_int(alpha=0.05)
            except Exception:
                ci = res.conf_int(alpha=0.05, method="wald")

            ci = np.asarray(ci)

            coef_feat = res.params[1]
            ci_lower  = ci[1, 0]
            ci_upper  = ci[1, 1]

            row = {
                "feature":  feature,
                "OR":       np.exp(coef_feat),
                "CI_lower": np.exp(ci_lower),
                "CI_upper": np.exp(ci_upper),
            }
            if feature_to_category is not None:
                row["category"] = feature_to_category[feature]
            uni_results.append(row)

        except Exception as e:
            print(f"  [ERROR] '{feature}' — model failed: {e}")

    if separated_features:
        print(
            f"\nFeatures dropped from univariate analysis due to perfect separation "
            f"({len(separated_features)}): {separated_features}"
        )

    uni_results = pd.DataFrame(uni_results)
    uni_results = _sort_grouped(uni_results, category_order)

    if feature_name_map is not None:
        uni_results["display_name"] = uni_results["feature"].map(
            lambda x: feature_name_map.get(x, x)
        )
    else:
        uni_results["display_name"] = uni_results["feature"]

    # ------------------------------------------------------------------
    # Reorder the multivariable results to match the univariate order.
    # Features dropped from the univariate analysis (perfect separation,
    # e.g. MedHist HF) have no univariate position, so each is inserted at
    # the end of its own category block (e.g. MedHist HF -> end of the
    # clinical features) rather than at the very end of the whole plot.
    # If no category grouping is active, separated features are simply
    # appended at the end, ordered by OR.
    # ------------------------------------------------------------------
    if category_order is not None:
        feature_order = []
        for cat in category_order:
            cat_features = list(uni_results.loc[uni_results["category"] == cat, "feature"])

            cat_separated = [f for f in separated_features if feature_to_category.get(f) == cat]
            if cat_separated:
                sep_sub = multi_results[multi_results["feature"].isin(cat_separated)].sort_values("OR")
                cat_features += list(sep_sub["feature"])

            feature_order += cat_features
    else:
        feature_order = list(uni_results["feature"])
        if separated_features:
            sep_df = multi_results[multi_results["feature"].isin(separated_features)].sort_values("OR")
            feature_order += list(sep_df["feature"])

    multi_results = (
        multi_results.set_index("feature").loc[feature_order].reset_index()
    )

    print("\nMultivariable Logistic Regression\n")
    print(multi_results[["display_name", "OR", "CI_lower", "CI_upper"]].round(3))

    _forest_plot(
        multi_results,
        title="Multivariate Logistic Regression",
        category_order=category_order,
        fontsize=fontsize,
        figsize=figsize,
        max_rows=max_rows,
    )

    print("\nUnivariate Logistic Regression\n")
    print(uni_results[["display_name", "OR", "CI_lower", "CI_upper"]].round(3))

    _forest_plot(
        uni_results,
        title="Univariate Logistic Regression",
        category_order=category_order,
        fontsize=fontsize,
        figsize=figsize,
        max_rows=max_rows,
    )

    return {
        "model":                  model,
        "selected_features":      selected_features,
        "multivariable_results":  multi_results,
        "univariate_results":     uni_results,
        "separated_features":     separated_features,
        "scaler":                 scaler,
        "feature_categories":     feature_to_category,
    }


def _forest_plot(results_df, title, category_order=None, fontsize=11, figsize=None, max_rows=20):
    sns.set(style="whitegrid")

    results_df = results_df.reset_index(drop=True)
    grouped = category_order is not None and "category" in results_df.columns
    n = len(results_df)

    # Row pitch (vertical space per row) is fixed by max_rows rather than
    # the actual number of features, so runs with fewer features simply
    # leave blank rows at the bottom instead of compressing/expanding
    # the spacing between rows.
    if figsize is None:
        figsize = (9, 0.45 * max_rows + 2)

    fig, (ax_forest, ax_text) = plt.subplots(
        ncols=2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1]},
        sharey=True,
    )

    y_pos = np.arange(n)

    # Plot each point individually so each can take its category's color
    for i, row in results_df.iterrows():
        color = CATEGORY_COLORS.get(row["category"], "black") if grouped else "black"
        ax_forest.errorbar(
            row["OR"],
            i,
            xerr=[[row["OR"] - row["CI_lower"]], [row["CI_upper"] - row["OR"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )

    ax_forest.set_yticks(y_pos)
    ax_forest.set_yticklabels(results_df["display_name"], fontsize=fontsize)
    # Fix the y-axis span to max_rows (instead of invert_yaxis(), which would
    # autoscale to just the actual rows) so row pitch stays identical
    # regardless of n; unused rows below the real features render as blank
    # whitespace.
    ax_forest.set_ylim(max_rows - 0.5, -0.5)
    ax_forest.axvline(1, linestyle="--", color="black")
    ax_forest.set_xscale("log")
    ax_forest.set_xlabel("Odds Ratio", fontsize=fontsize)
    ax_forest.set_title(title, fontsize=fontsize + 2)

    # Exactly 3 ticks: one below 1, 1 itself, one above 1 — chosen as "nice"
    # round values that bound the actual OR/CI range present in this plot,
    # so the spacing reflects how far the data actually spreads from 1.
    if n > 0:
        data_vals = pd.concat([
            results_df["OR"], results_df["CI_lower"], results_df["CI_upper"]
        ])
        vmin, vmax = data_vals.min(), data_vals.max()
    else:
        vmin, vmax = 1, 1

    lower_tick = _nice_floor(min(vmin, 1))
    if lower_tick >= 1:
        lower_tick = 0.5  # guarantee a tick below 1 even if no data dips below it

    upper_tick = _nice_ceil(max(vmax, 1))
    if upper_tick <= 1:
        upper_tick = 2  # guarantee a tick above 1 even if no data rises above it

    ax_forest.set_xticks([lower_tick, 1, upper_tick])
    ax_forest.xaxis.set_minor_locator(ticker.NullLocator())
    ax_forest.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax_forest.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    ax_forest.tick_params(axis="x", labelsize=fontsize)

    # Color the y-axis labels to match each feature's category
    if grouped:
        for tick_label, (_, row) in zip(ax_forest.get_yticklabels(), results_df.iterrows()):
            tick_label.set_color(CATEGORY_COLORS.get(row["category"], "black"))

    ax_text.yaxis.set_visible(False)
    ax_text.set_frame_on(False)

    for i, row in results_df.iterrows():
        label = f"{row.OR:.2f} [{row.CI_lower:.2f}–{row.CI_upper:.2f}]"
        ax_text.text(0, i, label, va="center", ha="left", fontsize=fontsize)

    ax_text.set_xlim(0, 1)
    ax_text.set_xticks([])

    plt.tight_layout()
    plt.show()