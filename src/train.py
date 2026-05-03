# Training dei modelli di classificazione (reparto + sentiment)
# con cross-validation, GridSearchCV e baseline comparative.

import os
import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)


RANDOM_STATE = 42
#Griglia per la ricerca degli iperparametri 
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.1, 1.0, 10.0],
}


def print_header(title):
    print("=" * 60)
    print(title)
    print("=" * 60)

# Caricamento del dataset 
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df["text"] = df["title"] + " " + df["body"]
    df["text"] = df["text"].apply(lambda x: re.sub(r"[^\w\s]", "", x))
    # Etichetta combinata per stratificare train/test su entrambi i target
    df["strat_label"] = df["department"] + "_" + df["sentiment"]
    print(f"Dataset caricato: {len(df)} recensioni")
    print(f"Distribuzione strat_label:\n{df['strat_label'].value_counts()}\n")
    return df

# Split per il train/test con stratificazione su reparto+sentiment
def split_data(df, test_size=0.2, seed=RANDOM_STATE):
    texts = df["text"]
    y_dept = df["department"]
    y_sent = df["sentiment"]
    strat = df["strat_label"]
    return train_test_split(
        texts, y_dept, y_sent,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

# Costruzione della pipeline con TF-IDF e Logistic Regression
def build_pipeline(class_weight=None):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        )),
    ])

# Funzione per eseguire il GridSearchCV e stampare i risultati
def tune_classifier(pipe, param_grid, texts_train, y_train, cv):
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(texts_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1 (macro): {grid.best_score_:.4f}")
    return grid

# Confronto con altre basline
def evaluate_baselines(X_train, y_train, cv, class_weight=None):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight=class_weight,
        ),
        "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(
            C=1.0, random_state=RANDOM_STATE, max_iter=2000,
            class_weight=class_weight,
        ),
    }
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="f1_macro", n_jobs=-1)
        results[name] = {
            "f1_mean": float(scores.mean()),
            "f1_std": float(scores.std()),
        }
        print(f"  {name:20s} F1 = {scores.mean():.4f} +- {scores.std():.4f}")
    return results

# Valutazione finale su test set e stampa dei risultati
def evaluate_on_test(grid, texts_test, y_test, label_name):
    y_pred = grid.predict(texts_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\nTest accuracy ({label_name}): {acc:.4f}")
    print(f"Test F1 macro ({label_name}): {f1m:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")
    return y_pred, acc, f1m

# Stampa di alcuni esempi di errori per un'analisi qualitativa
def print_error_examples(df_test, true_col, pred_col, title, n=3):
    errors = df_test[df_test[true_col] != df_test[pred_col]]
    print(f"\nErrori {title}: {len(errors)}/{len(df_test)}")
    for _, row in errors.head(n).iterrows():
        body = row["body"][:60]
        print(f"  \"{row['title']} - {body}...\"")
        print(f"  Vero: {row[true_col]} | Predetto: {row[pred_col]}")

# Funzioni per creare le matrici di confusione e i F1 score per classe
def plot_confusion_matrices(y_dept_test, dept_pred, y_sent_test, sent_pred,
                            dept_labels, sent_labels, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_dept = confusion_matrix(y_dept_test, dept_pred, labels=dept_labels)
    sns.heatmap(cm_dept, annot=True, fmt="d", cmap="Blues",
                xticklabels=dept_labels, yticklabels=dept_labels, ax=axes[0])
    axes[0].set_title("Confusion Matrix - Reparto")
    axes[0].set_xlabel("Predetto")
    axes[0].set_ylabel("Reale")

    cm_sent = confusion_matrix(y_sent_test, sent_pred, labels=sent_labels)
    sns.heatmap(cm_sent, annot=True, fmt="d", cmap="Oranges",
                xticklabels=sent_labels, yticklabels=sent_labels, ax=axes[1])
    axes[1].set_title("Confusion Matrix - Sentiment")
    axes[1].set_xlabel("Predetto")
    axes[1].set_ylabel("Reale")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_f1_per_class(y_dept_test, dept_pred, y_sent_test, sent_pred,
                      dept_labels, sent_labels, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dept_f1 = f1_score(y_dept_test, dept_pred, average=None, labels=dept_labels)
    axes[0].bar(dept_labels, dept_f1, color=["blue", "green", "orange"])
    axes[0].set_title("F1 Score per Reparto")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_ylim(0, 1.05)
    for i, v in enumerate(dept_f1):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    sent_f1 = f1_score(y_sent_test, sent_pred, average=None, labels=sent_labels)
    axes[1].bar(sent_labels, sent_f1, color=["red", "green"])
    axes[1].set_title("F1 Score per Sentiment")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(sent_f1):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

# Funzioni per salvare le metriche in un file JSON e i modelli addestrati
def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def save_models(grid_dept, grid_sent, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(grid_dept.best_estimator_.named_steps["clf"],
                os.path.join(models_dir, "department_model.pkl"))
    joblib.dump(grid_sent.best_estimator_.named_steps["clf"],
                os.path.join(models_dir, "sentiment_model.pkl"))
    joblib.dump(grid_dept.best_estimator_.named_steps["tfidf"],
                os.path.join(models_dir, "tfidf_dept.pkl"))
    joblib.dump(grid_sent.best_estimator_.named_steps["tfidf"],
                os.path.join(models_dir, "tfidf_sent.pkl"))

# Main, esegue l'intero processo di training, valutazione e salvataggio dei risultati
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root, "data", "reviews.csv")
    output_dir = os.path.join(root, "output")
    models_dir = os.path.join(root, "models")
    os.makedirs(output_dir, exist_ok=True)

    print_header("PIPELINE DI TRAINING")

    df = load_and_prepare_data(csv_path)
    texts_train, texts_test, y_dept_train, y_dept_test, y_sent_train, y_sent_test = split_data(df)
    print(f"Training: {len(texts_train)} | Test: {len(texts_test)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print_header("GRIDSEARCH REPARTO")
    grid_dept = tune_classifier(build_pipeline(), PARAM_GRID,
                                texts_train, y_dept_train, skf)

    print_header("GRIDSEARCH SENTIMENT")
    grid_sent = tune_classifier(build_pipeline(class_weight="balanced"),
                                PARAM_GRID, texts_train, y_sent_train, skf)

    print_header("BASELINE COMPARATIVE - REPARTO")
    X_tr_dept = grid_dept.best_estimator_.named_steps["tfidf"].transform(texts_train)
    baseline_dept = evaluate_baselines(X_tr_dept, y_dept_train, skf)

    print_header("BASELINE COMPARATIVE - SENTIMENT")
    X_tr_sent = grid_sent.best_estimator_.named_steps["tfidf"].transform(texts_train)
    baseline_sent = evaluate_baselines(X_tr_sent, y_sent_train, skf,
                                       class_weight="balanced")

    print_header("VALUTAZIONE FINALE SU TEST SET")
    dept_pred, dept_acc, dept_f1 = evaluate_on_test(
        grid_dept, texts_test, y_dept_test, "reparto",
    )
    sent_pred, sent_acc, sent_f1 = evaluate_on_test(
        grid_sent, texts_test, y_sent_test, "sentiment",
    )

    print_header("ESEMPI DI ERRORI")
    test_df = df.loc[y_dept_test.index].copy()
    test_df["dept_pred"] = dept_pred
    test_df["sent_pred"] = sent_pred
    print_error_examples(test_df, "department", "dept_pred", "reparto")
    print_error_examples(test_df, "sentiment", "sent_pred", "sentiment")

    dept_labels = sorted(df["department"].unique())
    sent_labels = sorted(df["sentiment"].unique())
    plot_confusion_matrices(
        y_dept_test, dept_pred, y_sent_test, sent_pred,
        dept_labels, sent_labels,
        os.path.join(output_dir, "confusion_matrix.png"),
    )
    plot_f1_per_class(
        y_dept_test, dept_pred, y_sent_test, sent_pred,
        dept_labels, sent_labels,
        os.path.join(output_dir, "f1_per_class.png"),
    )
    print(f"\nGrafici salvati in: {output_dir}/")

    metrics = {
        "dataset_size": len(df),
        "train_size": len(texts_train),
        "test_size": len(texts_test),
        "department": {
            "best_params": {k: str(v) for k, v in grid_dept.best_params_.items()},
            "cv_f1_mean": float(grid_dept.best_score_),
            "test_accuracy": float(dept_acc),
            "test_f1_macro": float(dept_f1),
            "baselines": baseline_dept,
        },
        "sentiment": {
            "best_params": {k: str(v) for k, v in grid_sent.best_params_.items()},
            "cv_f1_mean": float(grid_sent.best_score_),
            "test_accuracy": float(sent_acc),
            "test_f1_macro": float(sent_f1),
            "baselines": baseline_sent,
        },
    }
    save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
    print(f"Metriche salvate in: {output_dir}/metrics.json")

    save_models(grid_dept, grid_sent, models_dir)
    print(f"Modelli salvati in: {models_dir}/")
