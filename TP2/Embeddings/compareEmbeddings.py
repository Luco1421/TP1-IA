import gc
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from TP2 import DEVICE, Dataset, TrainRegression, train_test_split


RANDOM_STATE = 777
SEEDS = [101, 202, 303]
TEST_SIZE = 0.2
N_PAIRS = 15
N_TRIALS = 5
TRAIN_ITERATIONS = 500
EPSILON = 1e-5
FIXED_ALPHA = 0.01
BATCH_SIZE = 8

optuna.logging.set_verbosity(optuna.logging.WARNING)


dataset = Dataset("FEINA_1.xlsx").read()
dataset.corpus = dataset.corpus[:N_PAIRS * 2]
dataset.labels = dataset.labels[:N_PAIRS * 2]


def encode_transformer(model_name, corpus):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(corpus), BATCH_SIZE):
            batch = corpus[i:i + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            outputs = model(**encoded)
            batch_embeddings = outputs.pooler_output
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)


def encode_sentence_transformer(model_name, corpus):
    model = SentenceTransformer(model_name)
    model.to(DEVICE)
    embeddings = model.encode(
        corpus,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def split_data(matrix, labels, seed):
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    data_tensor = matrix.float()

    x_train, x_test, y_train, y_test = train_test_split(
        data_tensor,
        labels_tensor,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=labels,
    )

    x_train = x_train.to(DEVICE)
    x_test = x_test.to(DEVICE)
    y_train = y_train.to(DEVICE)
    y_test = y_test.to(DEVICE)

    return x_train, x_test, y_train, y_test


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
    return alpha_min, alpha_max, lipschitz


def calibrate_alpha(split, seed):
    alpha_min, alpha_max, _ = alpha_range(split)

    def objective(trial):
        alpha = trial.suggest_float("alpha", alpha_min, alpha_max, log=True)
        mae, _, _ = train_regression(split, alpha, seed)
        return mae

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params["alpha"], study.best_value


def test_embedding(name, model_name, encoder):
    print(f"\n{'=' * 100}")
    print(f"Embedding: {name}")
    print("Modelo:", model_name)

    embeddings = encoder(model_name, dataset.corpus)
    print("Documentos:", len(dataset.corpus))
    print("Dimensiones:", embeddings.shape[1])

    rows = []
    for seed in SEEDS:
        split = split_data(embeddings, dataset.labels, seed)
        alpha_min, alpha_max, lipschitz = alpha_range(split)
        fixed_mae, fixed_accuracy, _ = train_regression(split, FIXED_ALPHA, seed)
        tuned_alpha, _ = calibrate_alpha(split, seed)
        tuned_mae, tuned_accuracy, tuned_iterations = train_regression(split, tuned_alpha, seed)

        rows.append({
            "seed": seed,
            "lipschitz": round(lipschitz, 6),
            "alpha_min": round(alpha_min, 6),
            "alpha_max": round(alpha_max, 6),
            "alpha_fijo": FIXED_ALPHA,
            "mae_fijo": round(fixed_mae, 6),
            "accuracy_fijo": round(fixed_accuracy, 6),
            "alpha_calibrado": round(tuned_alpha, 6),
            "iteraciones": tuned_iterations,
            "mae_calibrado": round(tuned_mae, 6),
            "accuracy_calibrado": round(tuned_accuracy, 6),
        })

    print("\nResultados por seed:")
    print(pd.DataFrame(rows).to_string(index=False))

    return rows


embeddings = [
    (
        "actual TP2 distiluse multilingual",
        "sentence-transformers/distiluse-base-multilingual-cased-v1",
        encode_sentence_transformer,
    ),
    (
        "bert multilingual",
        "bert-base-multilingual-cased",
        encode_transformer,
    ),
    (
        "beto",
        "dccuchile/bert-base-spanish-wwm-cased",
        encode_transformer,
    ),
    (
        "roberta espanol",
        "PlanTL-GOB-ES/roberta-base-bne",
        encode_transformer,
    ),
]


def main():
    print("N_TRIALS:", N_TRIALS)

    rows = []
    for name, model_name, encoder in embeddings:
        test_embedding(name, model_name, encoder)


if __name__ == "__main__":
    main()
