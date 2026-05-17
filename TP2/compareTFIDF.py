import re
import nltk
import optuna
import pandas as pd
import simplemma
import torch
import spacy
import stanza
from nltk import RegexpTokenizer

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from TP2 import DEVICE, Dataset, TrainRegression, train_test_split


REGEX_NUMBERS = r"[^\W_]+"
REGEX_LETTERS = r"[a-záéíóúñü]+" # para no incluir numeros

RANDOM_STATE = 777
SEEDS = [101, 202, 303]
TEST_SIZE = 0.2
N_PAIRS = 15
N_TRIALS = 5
TRAIN_ITERATIONS = 500
EPSILON = 1e-5
FIXED_ALPHA = 0.01

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Descargar recursos de NLTK.
nltk.download("punkt", quiet=True) # Tokenizador
nltk.download("punkt_tab", quiet=True) # Tokenizador
nltk.download("stopwords", quiet=True) # Stopwords
nltk.download("wordnet", quiet=True) # Synsets
nltk.download("omw-1.4", quiet=True) # Lematizador

# Descargar recursos de Spacy.
spacy_nlp = spacy.load("es_core_news_lg")

# Descargar recursos de Stanza.
stanza_nlp = stanza.Pipeline(
    "es",
    processors="tokenize,lemma",
    use_gpu=True
)


dataset = Dataset("FEINA_1.xlsx").read()
dataset.corpus = dataset.corpus[:N_PAIRS * 2]
dataset.labels = dataset.labels[:N_PAIRS * 2]

# Stopwords
STOP_WORDS_ES = set(stopwords.words("spanish"))

# Tokenizadores
def tokenize_nltk(text, regex=None):
    return word_tokenize(text.lower(), language="spanish") # permite $ %

def tokenize_regex(text, regex):
    return RegexpTokenizer(regex).tokenize(text.lower())

# Lematizadores
def nltk_lemmatizer(token):
    synsets = wordnet.synsets(token, lang="spa")
    if synsets:
        lemmas = synsets[0].lemma_names("spa")
        if lemmas:
            return lemmas[0].lower().replace("_", " ")
    return token

def simplemma_lemmatizer(token):
    return simplemma.lemmatize(token, lang="es")


# Preprocesadores
def preprocess(text, tokenizer, lemmatizer, regex=None):
    tokens = tokenizer(text, regex)
    return [
        lemmatizer(token)
        for token in tokens
        if token not in STOP_WORDS_ES
    ]

def preprocess_spacy(text, mode):
    doc = spacy_nlp(text)
    lemmas = []
    for token in doc:
        lemma = token.lemma_.lower()
        if mode == 0:
            if (token.is_alpha or token.like_num) and not token.is_stop:
                lemmas.append(lemma)
        elif mode == 1: # permite $ %
            if not token.is_space and not token.is_stop:
                lemmas.append(lemma)
    return lemmas

def preprocess_stanza(text, mode, regex):
    doc = stanza_nlp(text)
    lemmas = []
    for sentence in doc.sentences:
        for token in sentence.words:
            lemma = token.lemma.lower()
            if mode == 0:
                if re.fullmatch(regex, lemma) and lemma not in STOP_WORDS_ES:
                    lemmas.append(lemma)
            elif mode ==1: # permite $ %
                if lemma not in STOP_WORDS_ES:
                    lemmas.append(lemma)
    return lemmas


def split_data(matrix, labels, seed):
    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    data_tensor = torch.tensor(matrix.toarray(), dtype=torch.float32)

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


def test_preprocessor(name, preprocessor):
    print(f"\n{'=' * 100}")
    print(f"Preprocesador: {name}")

    print("\nEjemplos de tokens:")
    corpus_tokens = [preprocessor(text) for text in dataset.corpus]
    for i, tokens in enumerate(corpus_tokens[:3], start=1):
        print(f"{i}. {tokens[:35]}")

    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens)
    matrix = vectorizer.fit_transform(corpus_tokens)
    tokens = vectorizer.get_feature_names_out()
    token_counts = [len(tokens) for tokens in corpus_tokens]

    print("\nResumen TF-IDF:")
    print("Documentos:", len(dataset.corpus))
    print("Vocabulario:", len(tokens))
    print("Promedio de tokens:", round(sum(token_counts) / len(token_counts), 3))
    print("Maximo de tokens:", max(token_counts))

    rows = []
    for seed in SEEDS:
        split = split_data(matrix, dataset.labels, seed)
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


preprocessors = [
    ("nltk modo nltk", lambda text: preprocess(text, tokenize_nltk, nltk_lemmatizer)),
    ("nltk modo regex con numeros", lambda text: preprocess(text, tokenize_regex, nltk_lemmatizer, REGEX_NUMBERS)),
    ("nltk modo regex sin numeros", lambda text: preprocess(text, tokenize_regex, nltk_lemmatizer, REGEX_LETTERS)),
    ("regex con numeros + simplemma", lambda text: preprocess(text, tokenize_regex, simplemma_lemmatizer, REGEX_NUMBERS)),
    ("regex sin numeros + simplemma", lambda text: preprocess(text, tokenize_regex, simplemma_lemmatizer, REGEX_LETTERS)),
    ("spacy lg modo estandar", lambda text: preprocess_spacy(text, 0)),
    ("spacy lg modo permisivo", lambda text: preprocess_spacy(text, 1)),
    ("stanza modo estandar con numeros", lambda text: preprocess_stanza(text, 0, REGEX_NUMBERS)),
    ("stanza modo estandar sin numeros", lambda text: preprocess_stanza(text, 0, REGEX_LETTERS)),
    ("stanza modo permisivo con numeros", lambda text: preprocess_stanza(text, 1, REGEX_NUMBERS)),
    ("stanza modo permisivo sin numeros", lambda text: preprocess_stanza(text, 1, REGEX_LETTERS))
]


def main():
    print("N_TRIALS:", N_TRIALS)

    rows = []
    for name, preprocessor in preprocessors:
        rows.extend(test_preprocessor(name, preprocessor))

if __name__ == "__main__":
    main()
