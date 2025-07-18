import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.stats import bernoulli
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


def get_morgan2(mol):
    return list(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048))


def compute_2Drdkit(mol):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    ds = calc.CalcDescriptors(mol)
    return list(ds)

def MLP_simulation_test(X_new,y):
    cv = StratifiedKFold(n_splits=10)
    MCCs = []
    for _i, (train, test) in enumerate(cv.split(X_new, y)):
        p_estimate = np.mean(y[train])
        n_test_samples = len(X_new[test])
        y_random = bernoulli.rvs(p_estimate, size=n_test_samples)
        mcc = matthews_corrcoef(y[test],y_random)
        MCCs.append(mcc)
    return np.mean(MCCs)


def MLP_decision_threshold_optimization(X, y, opt_num_feature):
    selector = SelectKBest(f_classif, k=opt_num_feature)
    X_new = selector.fit_transform(X, y)
    cv = StratifiedKFold(n_splits=10)
    classifier = MLPClassifier(
        hidden_layer_sizes=(10,),
        random_state=42,
        learning_rate_init=0.0001,
        max_iter=600,
    )
    opt_threshold_ap = []
    for _i, (train, test) in enumerate(cv.split(X_new, y)):
        classifier.fit(X_new[train], y[train])
        y_prob = classifier.predict_proba(X_new[test])[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y[test], y_prob)
        opt_id = np.argmax(tpr - fpr)
        opt_threshold = thresholds[opt_id]
        opt_threshold_ap.append(opt_threshold)
    opt_threshold_mean = np.mean(opt_threshold_ap)
    return opt_threshold_ap, opt_threshold_mean


def MLP_cv(X, y, opt_num_feature="all", threshold=None) -> None:
    selector = SelectKBest(f_classif, k=opt_num_feature)
    X_new = selector.fit_transform(X, y)
    cv = StratifiedKFold(n_splits=10)
    if opt_num_feature == "all" and threshold is None:
        classifier = MLPClassifier(random_state=42)
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(10,),
            random_state=42,
            learning_rate_init=0.0001,
            max_iter=600,
        )
    tprs, aucs, MCCs, cms, baccs, recalls, precisions = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X_new, y)):
        classifier.fit(X_new[train], y[train])
        loss_curve = classifier.loss_curve_
        iter = classifier.n_iter_

        y_prob = classifier.predict_proba(X_new[test])[:, 1]

        if threshold is None:  # Default MLP
            y_pred_opt = classifier.predict(X_new[test])
        else:  # Optimized MLP with custom threshold
            y_pred_opt = (y_prob >= threshold).astype(int)

        mcc = matthews_corrcoef(y[test], y_pred_opt)
        bacc = balanced_accuracy_score(y[test], y_pred_opt)
        cm = confusion_matrix(y[test], y_pred_opt, labels=classifier.classes_)
        rec = recall_score(y[test], y_pred_opt)
        prec = precision_score(y[test], y_pred_opt)

        MCCs.append(mcc)
        baccs.append(bacc)
        cms.append(cm)
        recalls.append(rec)
        precisions.append(prec)
        # ROC curve
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_new[test],
            y[test],
            name=f"ROC fold {i}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        # loss function curve
        iteration = np.linspace(0, iter, iter)
        ax1.plot(
            iteration,
            loss_curve,
            color="mediumslateblue",
            lw=1,
            alpha=0.5,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        MCCs.append(mcc)
        cms.append(cm)
        baccs.append(bacc)
        recalls.append(rec)
        precisions.append(prec)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    np.mean(MCCs)
    np.std(MCCs)
    stacked_matrices = np.stack(cms, axis=0)
    mean_cms = np.mean(stacked_matrices, axis=0)
    mean_cms = np.round(mean_cms).astype(
        int,
    )  # get the integer of the mean number
    np.mean(baccs)
    np.std(baccs)
    np.mean(recalls)
    np.std(recalls)
    np.mean(precisions)
    np.std(precisions)

    ax1.set(
        xlabel="Iteration",
        ylabel="Loss",
        title="Loss function curve",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Chance",
        alpha=0.8,
    )
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=rf"Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})",
        lw=2,
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC Curve",
    )
    ax.legend(loc="lower right")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=mean_cms,
        display_labels=classifier.classes_,
    )
    disp.plot(xticks_rotation="vertical")


def RF_cv(X, y, opt_num_feature="all") -> None:
    selector = SelectKBest(f_classif, k=opt_num_feature)
    X_new = selector.fit_transform(X, y)
    cv = StratifiedKFold(n_splits=10)
    if opt_num_feature == "all":
        classifier = RandomForestClassifier(
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_split=2,
            random_state=42,
            max_depth=30,
        )
    tprs = []
    aucs = []
    MCCs = []
    cms = []
    baccs = []
    recalls = []
    precisions = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X_new, y)):
        classifier.fit(X_new[train], y[train])
        pred = classifier.predict(X_new[test])
        mcc = matthews_corrcoef(y[test], pred)
        bacc = balanced_accuracy_score(y[test], pred)
        cm = confusion_matrix(y[test], pred, labels=classifier.classes_)
        rec = recall_score(y[test], pred)
        prec = precision_score(y[test], pred)
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_new[test],
            y[test],
            name=f"ROC fold {i}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        MCCs.append(mcc)
        cms.append(cm)
        baccs.append(bacc)
        recalls.append(rec)
        precisions.append(prec)

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Chance",
        alpha=0.8,
    )
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    np.mean(MCCs)
    np.std(MCCs)
    stacked_matrices = np.stack(cms, axis=0)
    mean_cms = np.mean(stacked_matrices, axis=0)
    np.mean(baccs)
    np.std(baccs)
    np.mean(recalls)
    np.std(recalls)
    np.mean(precisions)
    np.std(precisions)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=rf"Mean ROC (AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f})",
        lw=2,
        alpha=0.8,
    )
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="ROC Curve",
    )
    ax.legend(loc="lower right")
    plt.show()
    disp = ConfusionMatrixDisplay(
        confusion_matrix=mean_cms,
        display_labels=classifier.classes_,
    )
    disp.plot(xticks_rotation="vertical", values_format=".2f")
    plt.show()


def roc_cv_ANOVA_10fold(X, y, k_value, rf=False):
    selector = SelectKBest(f_classif, k=k_value)
    X_new = selector.fit_transform(X, y)
    cv = StratifiedKFold(n_splits=10)
    classifier = MLPClassifier(random_state=42)
    if rf:
        classifier = RandomForestClassifier(
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
        )
    aucs = []
    MCCs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for _i, (train, test) in enumerate(cv.split(X_new, y)):
        classifier.fit(X_new[train], y[train])
        pred = classifier.predict(X_new[test])
        mcc = matthews_corrcoef(y[test], pred)
        y_score = classifier.predict_proba(X_new[test])[:, 1]
        roc_auc = roc_auc_score(y[test], y_score)
        fpr, tpr, _ = metrics.roc_curve(y[test], y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)
        MCCs.append(mcc)
    return tprs, aucs, MCCs


def feature_selection(X, y, feature_list, name, k_list) -> None:
    MCC_of_different_ANOVA = []
    mean_mcc = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    c = plt.cm.rainbow(np.linspace(0, 1, 20))

    for i, k in enumerate(k_list):
        if k != "all" and int(k) <= len(feature_list):
            tprs, aucs, MCCs = roc_cv_ANOVA_10fold(np.array(X), y, int(k))
        elif k == "all":
            tprs, aucs, MCCs = roc_cv_ANOVA_10fold(np.array(X), y, k)
        else:
            MCC_of_different_ANOVA.append(0)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        mean_mcc = np.mean(MCCs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=c[i],
            label=rf"K={k}, ROC_AUC = {mean_auc:0.2f} $\pm$ {std_auc:0.2f}, MCC = {mean_mcc:0.2f}",
            lw=2,
            alpha=0.4,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Chance",
        alpha=0.8,
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=rf"ROC Curve ({name})",
    )
    ax.set_xlabel("False Positive Rate (Positive label = 1.0)")
    ax.set_ylabel("True Positive Rate (Positive label = 1.0)")
    ax.legend(loc="lower right", fontsize=7)
    plt.show()


def pairwise_tanimoto_similarity_closest_distance(smi_list, query_set_smi_list, fpe):
    default_value = -1
    results = []
    for my_smi in tqdm(query_set_smi_list, desc="Calculating similarities"):
        sim_results = fpe.similarity(my_smi, 0, n_workers=31)
        sim_dict = dict.fromkeys(smi_list, default_value)
        for idx, value in sim_results:
            p = smi_list[idx]
            sim_dict[p] = max(sim_dict[p], value)
        results.append({"smi": my_smi, "dict": sim_dict})
    temp_save = pd.DataFrame(results)
    temp_save[smi_list] = [list(value_1.values()) for value_1 in temp_save["dict"]]
    temp_save["closest_distance"] = temp_save[smi_list].max(axis=1)
    temp_save["closest_smi"] = temp_save[smi_list].idxmax(axis=1)
    return temp_save.drop(columns=["dict"])


def trialblazer_train(
    training_set,
    y,
    features,
    k,
):
    selector = SelectKBest(f_classif, k=k)
    X = training_set[features]
    X_new = selector.fit_transform(X, y)
    classifier = MLPClassifier(
        hidden_layer_sizes=(10,),
        random_state=42,
        learning_rate_init=0.0001,
        max_iter=600,
    )  # this classifier should be store somewhere so that it won't be trained every time
    classifier.fit(X_new, y)
    return classifier, selector


def trialblazer_func(
    classifier,
    selector,
    test_set,
    threshold,
    training_fpe,
    training_set,
    features,
    remove_MultiComponent_cpd=True,
):
    X_columns = features
    test_set_aligned = test_set.reindex(columns=X_columns, fill_value=0)
    X_test_ANO = selector.transform(test_set_aligned)

    y_prob = classifier.predict_proba(X_test_ANO)
    y_pred_opt = (y_prob[:, 1] >= threshold).astype(int)

    test_set_aligned["id"] = test_set["id"]
    test_set_aligned["SmilesForDropDu"] = test_set["SmilesForDropDu"]
    test_set_aligned["prediction"] = y_pred_opt
    test_set_aligned["pred_prob_positive"] = y_prob[:, 1]
    test_set_aligned["pred_prob_negative"] = y_prob[:, 0]

    # PrOCTOR score
    odd = test_set_aligned.pred_prob_negative / test_set_aligned.pred_prob_positive
    PrOCTOR_score = np.log2(odd)
    test_set_aligned["PrOCTOR_score"] = PrOCTOR_score
    test_set_aligned["PrOCTOR_score"] = test_set_aligned["PrOCTOR_score"].apply(
        lambda x: f"{x:.3f}",
    )
    predict_result_sim = test_set_aligned.iloc[:, -6:]
    predict_result_sim["PrOCTOR_score"] = predict_result_sim["PrOCTOR_score"].astype(
        float,
    )
    if remove_MultiComponent_cpd:
        predict_result_sim_remove_multi = predict_result_sim[
            ~predict_result_sim["id"].str.contains(r"\d+x\d+")
        ]  # Remove multi-component drugs if the toxicity of the compounds is uncertain
        predict_result_sim_remove_multi = predict_result_sim_remove_multi.sort_values(
            by="PrOCTOR_score",
            ascending=False,
            ignore_index=True,
        )
        predict_result_sim = predict_result_sim_remove_multi

    # Applicability domain

    similarity_closest_distance = pairwise_tanimoto_similarity_closest_distance(
        list(training_set["SmilesForDropDu"]),
        list(predict_result_sim["SmilesForDropDu"]),
        training_fpe,
    )
    # prediction output
    predict_result_sim = predict_result_sim.rename(columns={"SmilesForDropDu": "smi"})
    prediction_output = predict_result_sim.merge(
        similarity_closest_distance[["smi", "closest_distance", "closest_smi"]],
        how="left",
        on="smi",
    )
    prediction_output["prediction"] = prediction_output["prediction"].map(
        {0: "benign", 1: "toxic"},
    )

    # return (
    #     predict_result_sim,
    #     similarity_closest_distance[["smi", "closest_distance", "closest_smi"]],
    # )
    return prediction_output


def Trialblazer(
    training_set,
    y,
    features,
    test_set,
    threshold,
    k,
    training_fpe,
    remove_MultiComponent_cpd=True,
    classifier=None,
    selector=None,
):
    if classifier is None:
        classifier, selector = trialblazer_train(
            training_set=training_set, y=y, features=features, k=k,
        )
    return trialblazer_func(
        classifier=classifier,
        selector=selector,
        test_set=test_set,
        threshold=threshold,
        training_fpe=training_fpe,
        remove_MultiComponent_cpd=remove_MultiComponent_cpd,
        features=features,
        training_set=training_set,
    )
