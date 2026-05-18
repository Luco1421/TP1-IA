import os

import nltk
import optuna
import pandas as pd
import simplemma
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from TP2 import DEVICE, Dataset, TrainRegression, train_test_split


RUNS = 30
TEST_SIZE = 0.2
TRAIN_ITERATIONS = 500
EPSILON = 1e-5
FIXED_ALPHA = 0.1
N_TRIALS = 5
EMBEDDING_MODEL = "sentence-transformers/distiluse-base-multilingual-cased-v1"

optuna.logging.set_verbosity(optuna.logging.WARNING)
nltk.download("stopwords", quiet=True)

STOP_WORDS_ES = set(stopwords.words("spanish"))

dataset = Dataset("FEINA_1.xlsx").read()


def preprocess_document(text):
    tokens = re_tokenize(text)
    return [
        simplemma.lemmatize(token, lang="es")
        for token in tokens
        if token not in STOP_WORDS_ES
    ]


def re_tokenize(text):
    import re
    return re.findall(r"[a-záéíóúñü]+", text.lower())


def split_indices(labels, seed):
    indexes = list(range(len(labels)))
    train_indexes, test_indexes = train_test_split(
        indexes,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels,
    )
    return train_indexes, test_indexes


def train_regression(split, alpha, seed):
    x_train, x_test, y_train, y_test = split

    torch.manual_seed(seed)
    model = TrainRegression()
    _, history_error = model.train(
        [x_train, y_train],
        iterations=TRAIN_ITERATIONS,
        epsilon=EPSILON,
        alpha=alpha,
    )

    mae = model.medium_absolute_error([x_test, y_test]).item()
    accuracy = model.accuracy([x_test, y_test]).item()
    return mae, accuracy, len(history_error)


def alpha_range(split):
    x_train, _, _, _ = split
    data = TrainRegression.transformData(x_train)
    gram = (data.T @ data) / data.shape[0]
    lipschitz = torch.linalg.eigvalsh(gram).max().item() / 4
    alpha_max = 1 / lipschitz
    alpha_min = alpha_max / 1000
    return alpha_min, alpha_max


def calibrate_alpha(split, seed):
    alpha_min, alpha_max = alpha_range(split)

    def objective(trial):
        alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)
        mae, _, _ = train_regression(split, alpha, seed)
        return mae

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params["alpha"]


def make_tensor_split(x_train, x_test, y_train, y_test):
    x_train = x_train.float().to(DEVICE)
    x_test = x_test.float().to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(DEVICE)
    return x_train, x_test, y_train, y_test


def evaluate_split(treatment, split, seed, extra=None):
    fixed_mae, fixed_accuracy, _ = train_regression(split, FIXED_ALPHA, seed)
    tuned_alpha = calibrate_alpha(split, seed)
    tuned_mae, tuned_accuracy, iterations = train_regression(split, tuned_alpha, seed)

    row = {
        "seed": seed,
        "treatment": treatment,
        "alpha_fijo": FIXED_ALPHA,
        "mae_fijo": fixed_mae,
        "accuracy_fijo": fixed_accuracy,
        "alpha_calibrado": tuned_alpha,
        "mae_calibrado": tuned_mae,
        "accuracy_calibrado": tuned_accuracy,
        "iteraciones": iterations,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_tfidf(seed, train_indexes, test_indexes):
    train_corpus = [dataset.corpus[i] for i in train_indexes]
    test_corpus = [dataset.corpus[i] for i in test_indexes]
    train_labels = [dataset.labels[i] for i in train_indexes]
    test_labels = [dataset.labels[i] for i in test_indexes]

    corpus_tokens_train = [preprocess_document(text) for text in train_corpus]
    corpus_tokens_test = [preprocess_document(text) for text in test_corpus]

    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens)
    x_train = vectorizer.fit_transform(corpus_tokens_train)
    x_test = vectorizer.transform(corpus_tokens_test)

    split = make_tensor_split(
        torch.tensor(x_train.toarray(), dtype=torch.float32),
        torch.tensor(x_test.toarray(), dtype=torch.float32),
        train_labels,
        test_labels,
    )

    return evaluate_split(
        "tfidf",
        split,
        seed,
        {"vocabulario": len(vectorizer.get_feature_names_out())},
    )


def make_embeddings(corpus):
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.to(DEVICE)
    return model.encode(
        corpus,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )


def evaluate_embeddings(seed, embeddings, train_indexes, test_indexes):
    train_labels = [dataset.labels[i] for i in train_indexes]
    test_labels = [dataset.labels[i] for i in test_indexes]

    split = make_tensor_split(
        embeddings[train_indexes],
        embeddings[test_indexes],
        train_labels,
        test_labels,
    )

    return evaluate_split(
        "embeddings",
        split,
        seed,
        {"modelo_embeddings": EMBEDDING_MODEL, "dimension": embeddings.shape[1]},
    )


def print_summary(rows):
    data = pd.DataFrame(rows)
    summary = (
        data
        .groupby("treatment")[["mae_calibrado", "accuracy_calibrado"]]
        .agg(["mean", "std"])
        .round(6)
    )
    print("\nResumen 30 corridas:")
    print(summary.to_string())


def save_results(rows, partitions):
    os.makedirs("output", exist_ok=True)
    pd.DataFrame(rows).to_csv("output/final_30_lr_results.csv", index=False)
    pd.DataFrame(partitions).to_csv("output/final_30_partitions.csv", index=False)
    print("\nArchivos generados:")
    print("output/final_30_lr_results.csv")
    print("output/final_30_partitions.csv")


def main():
    print("RUNS:", RUNS)
    print("TEST_SIZE:", TEST_SIZE)
    print("N_TRIALS:", N_TRIALS)

    embeddings = make_embeddings(dataset.corpus)

    rows = []
    partitions = []
    for seed in range(RUNS):
        print(f"\n{'=' * 100}")
        print("Seed:", seed)

        train_indexes, test_indexes = split_indices(dataset.labels, seed)
        partitions.append({
            "seed": seed,
            "train_indexes": " ".join(map(str, train_indexes)),
            "test_indexes": " ".join(map(str, test_indexes)),
        })
        print("Test indexes:", test_indexes)

        tfidf_row = evaluate_tfidf(seed, train_indexes, test_indexes)
        rows.append(tfidf_row)
        print("TF-IDF:", tfidf_row)

        embeddings_row = evaluate_embeddings(seed, embeddings, train_indexes, test_indexes)
        rows.append(embeddings_row)
        print("Embeddings:", embeddings_row)

    print_summary(rows)
    save_results(rows, partitions)


if __name__ == "__main__":
    main()
