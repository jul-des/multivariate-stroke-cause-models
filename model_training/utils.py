
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

def train_model_nested_cv(
    X: "pd.DataFrame",
    y: "pd.Series",
    model_type: str = "rf",
    feature_method: str = "mrmr",
    param_grid: dict = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    target_sensitivity: float = 0.75,
    include_treatment=False,
    toast_classes=[3]
):
   
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    if feature_method == "mrmr":
        try:
            from mrmr import mrmr_classif
        except ImportError:
            raise ImportError("Install mrmr-selection: pip install mrmr-selection")
    elif feature_method != "rfe":
        raise ValueError("feature_method must be 'mrmr' or 'rfe'")

    assert "Patient ID" in X.columns, "X must contain column 'Patient ID'"

    LABELS = toast_classes

    rng = np.random.RandomState(random_state)

    # -------------------------------------------------------
    # Default parameter grids
    # -------------------------------------------------------
    if param_grid is None:
        if model_type == "rf":
            param_grid = {
                "n_estimators": [200, 500],
                "max_depth": [3, 5, 10],
                "min_samples_leaf": [1, 2, 5],
            }
        elif model_type == "xgb":
            param_grid = {
                "n_estimators": [100, 300],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.7, 1.0],
            }
        elif model_type == "svm":

            param_grid = {
                "kernel": ["linear", "rbf"],
                "C": [0.01, 0.1, 1, 10],
                "gamma": ["scale", 0.001, 0.01, 0.1, 1]
            }


        elif model_type == "logreg":
            param_grid = {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["liblinear"],
            }
        else:
            raise ValueError("model_type must be one of: 'rf', 'xgb', 'svm', 'logreg'")

    param_grid_with_nf = {**param_grid, "n_features": [10, 15, 20]}
    param_list = list(ParameterGrid(param_grid_with_nf))

    # -------------------------------------------------------
    # Helper: instantiate classifier (binary for all OvR models)
    # -------------------------------------------------------
    def build_model(params: dict):
        p = {k: v for k, v in params.items() if k != "n_features"}
        if model_type == "rf":
            return RandomForestClassifier(
                class_weight="balanced", random_state=random_state, **p
            )
        elif model_type == "svm":
            return SVC(
                probability=True, class_weight="balanced",
                random_state=random_state, **p
            )
        elif model_type == "xgb":
            if XGBClassifier is None:
                raise ImportError("XGBoost not installed: pip install xgboost")
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                n_jobs=4,
                random_state=random_state,
                **p
            )
        elif model_type == "logreg":
            return LogisticRegression(
                class_weight="balanced", random_state=random_state,
                max_iter=1000, **p
            )

    # -------------------------------------------------------
    # Helper: fixed LogisticRegression estimator for RFE
    # -------------------------------------------------------
    def build_rfe_estimator():
        return LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear",
            class_weight="balanced", random_state=random_state,
            max_iter=1000
        )

    # -------------------------------------------------------
    # Helper: feature selection on binarised OvR target
    # -------------------------------------------------------
    def select_features(X_df: "pd.DataFrame", y_bin: "pd.Series", k: int) -> list:
        if feature_method == "mrmr":
            return mrmr_classif(X=X_df, y=y_bin, K=k)
        else:  # rfe
            estimator = build_rfe_estimator()
            selector  = RFE(estimator=estimator, n_features_to_select=k, step=1)
            selector.fit(X_df.values, y_bin.values)
            return list(X_df.columns[selector.support_])

    # -------------------------------------------------------
    # Helper: cutoff whose sensitivity (TPR) is closest to
    # `target_sensitivity` on the pooled ROC curve. Ties (multiple
    # thresholds equally close to the target) are broken by picking
    # the one with the highest specificity (1 - FPR).
    # -------------------------------------------------------
    def sensitivity_target_cutoff(y_true, y_probs, target=target_sensitivity):
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        diffs = np.abs(tpr - target)
        min_diff = diffs.min()
        candidate_idx = np.where(diffs == min_diff)[0]
        # Tie-break: among equally-close candidates, prefer highest specificity
        best_idx = candidate_idx[np.argmax(1.0 - fpr[candidate_idx])]
        cutoff      = thresholds[best_idx]
        sensitivity = tpr[best_idx]
        specificity = 1.0 - fpr[best_idx]
        return cutoff, sensitivity, specificity

    # -------------------------------------------------------
    # Helper: sensitivity/specificity of predictions vs labels
    # at a fixed cutoff (used during bootstrap resampling)
    # -------------------------------------------------------
    def sens_spec_at_cutoff(y_true, y_probs, cutoff):
        preds = (y_probs >= cutoff).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        fp = np.sum((preds == 1) & (y_true == 0))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return sens, spec

    # -------------------------------------------------------
    # Main loop: one full nested CV pipeline per OvR class
    # -------------------------------------------------------
    all_results = {}

    for target_class in LABELS:

        print(f"\n{'#'*55}")
        print(f"OvR MODEL: class {target_class} vs rest")
        print(f"{'#'*55}")

        # Binarise target: 1 if this class, 0 otherwise
        y_bin = (y == target_class).astype(int)

        fold_auc_scores      = []
        best_params_per_fold = []
        features_per_fold    = []
        pooled_probs         = []
        pooled_labels        = []

        # -------------------------------------------------------
        # Outer fold loop
        # -------------------------------------------------------
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        outer_splits = list(outer_cv.split(X, y))
        for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):

            print(f"\n{'='*55}")
            print(f"  OUTER FOLD {fold}  |  class {target_class} vs rest")
            print(f"{'='*55}")

            X_outer_train_full = X.iloc[outer_train_idx].drop(columns=["Patient ID"])
            y_outer_train_bin  = y_bin.iloc[outer_train_idx]

            X_outer_test_full = X.iloc[outer_test_idx].drop(columns=["Patient ID"])
            y_outer_test_bin  = y_bin.iloc[outer_test_idx]

            # -------------------------------------------------------
            # Inner loop: StratifiedKFold on outer train
            # Stratify on binarised target to preserve positive class ratio
            # -------------------------------------------------------
            inner_cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            inner_splits = list(inner_cv.split(X_outer_train_full, y_outer_train_bin))

            param_inner_aucs = np.zeros(len(param_list))

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):

                X_inner_train_full = X_outer_train_full.iloc[inner_train_idx]
                y_inner_train_bin  = y_outer_train_bin.iloc[inner_train_idx]
                X_inner_val_full   = X_outer_train_full.iloc[inner_val_idx]
                y_inner_val_bin    = y_outer_train_bin.iloc[inner_val_idx]

                feature_cache: dict[int, list] = {}

                for i, params in enumerate(param_list):
                    k = params["n_features"]

                    if k not in feature_cache:
                        feature_cache[k] = select_features(
                            X_inner_train_full, y_inner_train_bin, k
                        )
                    sel_feats = feature_cache[k]
                    if include_treatment:
                        if "IAT" not in sel_feats:
                            sel_feats = sel_feats + ['IAT']
                        if "IVT with rtPA" not in sel_feats:
                            sel_feats = sel_feats + ['IVT with rtPA']

                    X_it = X_inner_train_full[sel_feats]
                    X_iv = X_inner_val_full[sel_feats]

                    scaler   = StandardScaler()
                    X_it_s   = scaler.fit_transform(X_it)
                    X_iv_s   = scaler.transform(X_iv)

                    model     = build_model(params)
                    model.fit(X_it_s, y_inner_train_bin)
                    val_probs = model.predict_proba(X_iv_s)[:, 1]
                    param_inner_aucs[i] += roc_auc_score(y_inner_val_bin, val_probs)

                print(f"    Inner fold {inner_fold}/5 complete")

            param_inner_aucs /= 5

            best_param_idx  = int(np.argmax(param_inner_aucs))
            best_params     = param_list[best_param_idx]
            best_n_features = best_params["n_features"]
            best_inner_auc  = param_inner_aucs[best_param_idx]

            best_params_per_fold.append(best_params)
            print(f"  Best inner CV AUC : {best_inner_auc:.4f}")
            print(f"  Best params       : {best_params}")

            # -------------------------------------------------------
            # Feature selection + scaling on full outer train
            # -------------------------------------------------------
            selected_features = select_features(
                X_outer_train_full, y_outer_train_bin, best_n_features
            )

            if include_treatment:
                if "IAT" not in selected_features:
                    selected_features = selected_features + ['IAT']
                if "IVT with rtPA" not in selected_features:
                    selected_features = selected_features + ['IVT with rtPA']

            features_per_fold.append(selected_features)
            print(f"  Selected features ({feature_method}, k={best_n_features}): {selected_features}")

            X_outer_train = X_outer_train_full[selected_features]
            X_outer_test  = X_outer_test_full[selected_features]

            scaler = StandardScaler()
            X_outer_train_s = scaler.fit_transform(X_outer_train)
            X_outer_test_s  = scaler.transform(X_outer_test)

            # -------------------------------------------------------
            # Retrain final model on full outer train
            # -------------------------------------------------------
            final_model = build_model(best_params)
            final_model.fit(X_outer_train_s, y_outer_train_bin)

            # -------------------------------------------------------
            # Evaluate on outer test set
            # -------------------------------------------------------
            test_probs = final_model.predict_proba(X_outer_test_s)[:, 1]
            test_auc   = roc_auc_score(y_outer_test_bin, test_probs)

            fold_auc_scores.append(test_auc)
            pooled_probs.extend(test_probs)
            pooled_labels.extend(y_outer_test_bin)

            print(f"  Outer fold {fold} test ROC-AUC: {test_auc:.4f}")

        # -------------------------------------------------------
        # Overall pooled ROC-AUC for this OvR model
        # -------------------------------------------------------
        pooled_probs_arr  = np.array(pooled_probs)
        pooled_labels_arr = np.array(pooled_labels)

        overall_auc = roc_auc_score(pooled_labels_arr, pooled_probs_arr)

        # -------------------------------------------------------
        # Cutoff whose sensitivity is closest to `target_sensitivity`,
        # computed on the pooled outer-test predictions (single cutoff
        # per class), plus the sensitivity/specificity achieved there
        # -------------------------------------------------------
        cutoff, sensitivity, specificity = sensitivity_target_cutoff(
            pooled_labels_arr, pooled_probs_arr
        )

        # -------------------------------------------------------
        # Bootstrap 95% CI (AUC, and sensitivity/specificity at cutoff)
        # -------------------------------------------------------
        bootstrap_scores = []
        bootstrap_sens    = []
        bootstrap_spec    = []
        n = len(pooled_labels_arr)

        for _ in range(n_bootstrap):
            idx   = rng.choice(np.arange(n), size=n, replace=True)
            y_bs  = pooled_labels_arr[idx]
            p_bs  = pooled_probs_arr[idx]

            bootstrap_scores.append(roc_auc_score(y_bs, p_bs))

            sens_bs, spec_bs = sens_spec_at_cutoff(y_bs, p_bs, cutoff)
            bootstrap_sens.append(sens_bs)
            bootstrap_spec.append(spec_bs)

        ci_lower = float(np.percentile(bootstrap_scores, 2.5))
        ci_upper = float(np.percentile(bootstrap_scores, 97.5))

        sens_ci_lower = float(np.nanpercentile(bootstrap_sens, 2.5))
        sens_ci_upper = float(np.nanpercentile(bootstrap_sens, 97.5))
        spec_ci_lower = float(np.nanpercentile(bootstrap_spec, 2.5))
        spec_ci_upper = float(np.nanpercentile(bootstrap_spec, 97.5))

        print(f"\n{'='*55}")
        print(f"OvR RESULTS: class {target_class} vs rest")
        print(f"{'='*55}")
        for i, auc in enumerate(fold_auc_scores, 1):
            print(f"  Outer fold {i} test AUC: {auc:.4f}")
        print(f"\n  Overall pooled ROC-AUC : {overall_auc:.4f}")
        print(f"  95% Bootstrap CI       : [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Pooled cutoff (target sensitivity={target_sensitivity}): {cutoff:.4f}")
        print(f"  Sensitivity            : {sensitivity:.4f}  "
              f"95% CI [{sens_ci_lower:.4f}, {sens_ci_upper:.4f}]")
        print(f"  Specificity            : {specificity:.4f}  "
              f"95% CI [{spec_ci_lower:.4f}, {spec_ci_upper:.4f}]")

        all_results[target_class] = {
            "fold_auc_scores":      fold_auc_scores,
            "overall_auc":          overall_auc,
            "bootstrap_ci":         (ci_lower, ci_upper),
            "best_params_per_fold": best_params_per_fold,
            "features_per_fold":    features_per_fold,
            "test_probabilities":   pooled_probs_arr,
            "test_labels":          pooled_labels_arr,
            "cutoff":               cutoff,
            "sensitivity":          sensitivity,
            "specificity":          specificity,
            "sensitivity_ci":       (sens_ci_lower, sens_ci_upper),
            "specificity_ci":       (spec_ci_lower, spec_ci_upper),
        }

    # -------------------------------------------------------
    # Weighted average ROC-AUC / sensitivity / specificity across
    # all 4 OvR models. Weights = class frequency in the full dataset
    # -------------------------------------------------------
    class_counts = y.value_counts()
    total        = class_counts.sum()
    weights      = {cls: class_counts.get(cls, 0) / total for cls in LABELS}

    weighted_avg_auc = sum(
        weights[cls] * all_results[cls]["overall_auc"]
        for cls in LABELS
    )
    weighted_avg_sensitivity = sum(
        weights[cls] * all_results[cls]["sensitivity"]
        for cls in LABELS
    )
    weighted_avg_specificity = sum(
        weights[cls] * all_results[cls]["specificity"]
        for cls in LABELS
    )

    # Bootstrap CI for the weighted averages: combine per-class bootstrap
    # distributions using the same weights. Each class uses a freshly seeded
    # RandomState (same seed) so bootstrap indices are reproducible per class,
    # and the same resamples are reused for AUC, sensitivity and specificity.
    bootstrap_weighted_auc  = np.zeros(n_bootstrap)
    bootstrap_weighted_sens = np.zeros(n_bootstrap)
    bootstrap_weighted_spec = np.zeros(n_bootstrap)
    for cls in LABELS:
        pooled_probs_arr  = all_results[cls]["test_probabilities"]
        pooled_labels_arr = all_results[cls]["test_labels"]
        cls_cutoff         = all_results[cls]["cutoff"]
        n                  = len(pooled_labels_arr)
        rng_ci             = np.random.RandomState(random_state + cls)

        per_class_auc_bs  = []
        per_class_sens_bs = []
        per_class_spec_bs = []
        for _ in range(n_bootstrap):
            idx  = rng_ci.choice(np.arange(n), size=n, replace=True)
            y_bs = pooled_labels_arr[idx]
            p_bs = pooled_probs_arr[idx]

            per_class_auc_bs.append(roc_auc_score(y_bs, p_bs))

            sens_bs, spec_bs = sens_spec_at_cutoff(y_bs, p_bs, cls_cutoff)
            per_class_sens_bs.append(sens_bs)
            per_class_spec_bs.append(spec_bs)

        bootstrap_weighted_auc  += weights[cls] * np.array(per_class_auc_bs)
        bootstrap_weighted_sens += weights[cls] * np.array(per_class_sens_bs)
        bootstrap_weighted_spec += weights[cls] * np.array(per_class_spec_bs)

    wa_ci_lower = float(np.percentile(bootstrap_weighted_auc, 2.5))
    wa_ci_upper = float(np.percentile(bootstrap_weighted_auc, 97.5))

    sens_wa_ci_lower = float(np.nanpercentile(bootstrap_weighted_sens, 2.5))
    sens_wa_ci_upper = float(np.nanpercentile(bootstrap_weighted_sens, 97.5))
    spec_wa_ci_lower = float(np.nanpercentile(bootstrap_weighted_spec, 2.5))
    spec_wa_ci_upper = float(np.nanpercentile(bootstrap_weighted_spec, 97.5))

    # -------------------------------------------------------
    # Final summary across all 4 OvR models
    # -------------------------------------------------------
    print(f"\n{'#'*55}")
    print("FINAL SUMMARY — ALL OvR MODELS")
    print(f"{'#'*55}")
    for cls in LABELS:
        auc      = all_results[cls]["overall_auc"]
        lo, hi   = all_results[cls]["bootstrap_ci"]
        sens     = all_results[cls]["sensitivity"]
        sens_lo, sens_hi = all_results[cls]["sensitivity_ci"]
        spec     = all_results[cls]["specificity"]
        spec_lo, spec_hi = all_results[cls]["specificity_ci"]
        w        = weights[cls]
        print(f"  Class {cls} vs rest  |  weight: {w:.3f}  AUC: {auc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"      Sensitivity: {sens:.4f}  95% CI [{sens_lo:.4f}, {sens_hi:.4f}]  |  "
              f"Specificity: {spec:.4f}  95% CI [{spec_lo:.4f}, {spec_hi:.4f}]")
    print(f"\n  Weighted avg AUC          : {weighted_avg_auc:.4f}  95% CI [{wa_ci_lower:.4f}, {wa_ci_upper:.4f}]")
    print(f"  Weighted avg sensitivity  : {weighted_avg_sensitivity:.4f}  "
          f"95% CI [{sens_wa_ci_lower:.4f}, {sens_wa_ci_upper:.4f}]")
    print(f"  Weighted avg specificity  : {weighted_avg_specificity:.4f}  "
          f"95% CI [{spec_wa_ci_lower:.4f}, {spec_wa_ci_upper:.4f}]")

    all_results["weighted_avg"] = {
        "overall_auc":     weighted_avg_auc,
        "bootstrap_ci":    (wa_ci_lower, wa_ci_upper),
        "sensitivity":     weighted_avg_sensitivity,
        "sensitivity_ci":  (sens_wa_ci_lower, sens_wa_ci_upper),
        "specificity":     weighted_avg_specificity,
        "specificity_ci":  (spec_wa_ci_lower, spec_wa_ci_upper),
        "weights":         weights,
    }

    return all_results


def save_results(results, all_results, filename, model, data, toast_categories):


    all_results["Model"].append(model)
    all_results["Data"].append(data)

    for i in [toast_categories]:
        if "AUC_" + str(i) not in all_results:
            all_results["AUC_" + str(i)] = []
            all_results["95%_lower_" + str(i)] = []
            all_results["95%_upper_" + str(i)] = []
            all_results["Cutoff_" + str(i)] = []
            all_results["Sensitivity_" + str(i)] = []
            all_results["Sens_95%_lower_" + str(i)] = []
            all_results["Sens_95%_upper_" + str(i)] = []
            all_results["Specificity_" + str(i)] = []
            all_results["Spec_95%_lower_" + str(i)] = []
            all_results["Spec_95%_upper_" + str(i)] = []

        all_results["95%_upper_" +  str(i)].append(results[i]["bootstrap_ci"][1])
        all_results["AUC_" + str(i)].append(results[i]["overall_auc"])
        all_results["95%_lower_" + str(i)].append(results[i]["bootstrap_ci"][0])

        all_results["Cutoff_" + str(i)].append(results[i]["cutoff"])
        all_results["Sensitivity_" + str(i)].append(results[i]["sensitivity"])
        all_results["Sens_95%_lower_" + str(i)].append(results[i]["sensitivity_ci"][0])
        all_results["Sens_95%_upper_" + str(i)].append(results[i]["sensitivity_ci"][1])
        all_results["Specificity_" + str(i)].append(results[i]["specificity"])
        all_results["Spec_95%_lower_" + str(i)].append(results[i]["specificity_ci"][0])
        all_results["Spec_95%_upper_" + str(i)].append(results[i]["specificity_ci"][1])

    if "AUC_overall" not in all_results:
            all_results["AUC_overall"] = []
            all_results["95%_lower_overall"] = []
            all_results["95%_upper_overall"] = []
            all_results["Sensitivity_overall"] = []
            all_results["Sens_95%_lower_overall"] = []
            all_results["Sens_95%_upper_overall"] = []
            all_results["Specificity_overall"] = []
            all_results["Spec_95%_lower_overall"] = []
            all_results["Spec_95%_upper_overall"] = []

    all_results["95%_upper_overall"].append(results["weighted_avg"]["bootstrap_ci"][1])
    all_results["AUC_overall"].append(results["weighted_avg"]["overall_auc"])
    all_results["95%_lower_overall"].append(results["weighted_avg"]["bootstrap_ci"][0]) 

    all_results["Sensitivity_overall"].append(results["weighted_avg"]["sensitivity"])
    all_results["Sens_95%_lower_overall"].append(results["weighted_avg"]["sensitivity_ci"][0])
    all_results["Sens_95%_upper_overall"].append(results["weighted_avg"]["sensitivity_ci"][1])
    all_results["Specificity_overall"].append(results["weighted_avg"]["specificity"])
    all_results["Spec_95%_lower_overall"].append(results["weighted_avg"]["specificity_ci"][0])
    all_results["Spec_95%_upper_overall"].append(results["weighted_avg"]["specificity_ci"][1])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(filename)  
    return all_results
