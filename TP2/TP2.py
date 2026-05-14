# %%
import sys, os

os.system(f'{sys.executable} -m pip install pip --upgrade')
os.system(f'{sys.executable} -m pip install torch scikit-learn matplotlib pandas nltk simplemma sentence-transformers transformers openpyxl')
# %% [markdown]
# # Imports
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_blobs
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
from nltk.corpus import stopwords
from torch.nn import Sigmoid

import matplotlib.pyplot as plt
import pandas as pd
import simplemma
import random
import torch
import nltk
import re
# %% [markdown]
# # Enviroment Variables
# %%
os.environ["HF_TOKEN"] = "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"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RELATIVE_ERROR = 1e-12

pd.options.display.float_format = (lambda x: '0' if x == 0 else f'{x}')
# %% [markdown]
# # Utils
# %%
class Visualization :

    def __init__(self):
        self.n = 0

    def make_scatter(self,x, y, c):
        plt.figure()

        plt.scatter(x, y, c=c)

        plt.title("Diagrama de Dipersión")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")

        plt.tight_layout()
        plt.show()
        self.save()

    def make_error_plot(self,history_error, title : str):
        plt.figure()

        plt.plot(history_error)
        plt.title(f"{title} - Gráfica de Error")
        plt.xlabel("Iteración")
        plt.ylabel("Error")
        plt.show()
        self.save()

    def make_scatter_with_line(self,x, y, c, y_line, title : str):
        plt.figure()
        plt.scatter(x,y,c=c)

        plt.plot(x,y_line)

        plt.title(f"Diagrama de Disperción {title} con la superficie de decisión")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.show()
        self.save()

    def save(self):
        plt.savefig(f"{self.n}.png")
        self.n += 1

visualization = Visualization()

class Table:
    """! The Table class
    Defines the functions used to generate, update and show a pandas dataframe
    """

    def __init__(self, data: pd.DataFrame, names : list[str], table_name : str):
        self.data = data

        self.table_name = table_name

        for name in names:
            data[name] = pd.Series(dtype=object)

    def add(self, row : int, column : str, value):
        """! Add a value in a specific column of the table
        @param row Row in the table
        @param column Column in the table
        @param value Value to add in the table
        """

        # Rounds float values for consistency
        if isinstance(value, float):
            value = round(value, 6)

        # Inserts or updates the value in the table
        self.data.loc[row, column] = value

    def show(self):
        """! Show the table
        """

        # Show the dataframe
        print(self.data)

    def generate_latex(self, index : bool):
        self.data.to_latex(f"{self.table_name}.tex", index=index)
# %% [markdown]
# # Perceptron Algorithm
# %%
class Perceptron:

    def forward(self, X : torch.Tensor, w : torch.Tensor):
        return torch.sign(X @ w)

    def gradient(self, X : torch.Tensor, w : torch.Tensor, t : torch.Tensor):
        perceptron_criteria = (X @ w) * t
        mask = (perceptron_criteria <= 0).squeeze()

        self.U = X[mask]
        self.t_U = t[mask]

        return (self.U * self.t_U).sum(dim=0).view(-1, 1)

    def error(self, w : torch.Tensor):
        return -((self.U @ w) * self.t_U).sum()

    def accuracy(self, X : torch.Tensor, w : torch.Tensor, t : torch.Tensor):
        return (self.forward(X, w) == t).float().mean()

    def train(self, X : torch.Tensor, w : torch.Tensor, t : torch.Tensor, iterations : int, alpha : float):
        history_error = []
        for i in range(iterations):
            history_error.append(self.error(w).item())
            delta = self.gradient(X, w, t)
            w = w + alpha * delta

        return w, history_error
# %% [markdown]
# # Logistic Regression
# %%
class LogisticRegression:

    def __init__(self):
        self.sigmoid = Sigmoid()

    def forward(self, X : torch.Tensor, w : torch.Tensor):
        return self.sigmoid(X @ w)

    def gradient(self, X :torch.Tensor, w : torch.Tensor, t : torch.Tensor):
        diff = (X * (t - self.forward(X, w))).sum(dim=0)
        return diff.view(-1, 1) / X.shape[0]

    def medium_absolute_error(self, X : torch.Tensor, w : torch.Tensor, t : torch.Tensor):
        return (t - self.forward(X, w)).abs().mean()

    def accuracy(self, X : torch.Tensor, w : torch.Tensor, t : torch.Tensor):
        return (self.forward(X, w).round() == t).float().mean()

    def train(self, X : torch.Tensor, w : torch.Tensor, t : torch.Tensor, iterations : int, alpha : float, epsilon: float):
        history_error = []
        for i in range(iterations):
            history_error.append(self.medium_absolute_error(X, w, t).item())
            delta = self.gradient(X, w, t)
            w = w + alpha * delta
            if i > 100 and delta.norm() < epsilon:
                break
        return w, history_error
# %% [markdown]
# # Unit tests: Forward
# %% [markdown]
# ## Test 1 - Forward
# %%
def logistic_regression_forward_test_1():
    # objetivos
    X = torch.tensor([
        [1.0, 35.0, 50],
        [1.0, 40.0, 53],
        [1.0, 25.0, 80],
        [1.0, 28.0, 73]
    ])

    w = torch.tensor([
        [0.3],
        [0.2],
        [0.25]
    ])

    expected_product = torch.tensor([
            [19.8],
            [21.55],
            [25.3],
            [24.15]
    ])

    sigmoid = Sigmoid()
    logistic_regression = LogisticRegression()

    expected_result = sigmoid(expected_product)
    result = logistic_regression.forward(X, w)

    assert (expected_result-result).abs().max().item() < RELATIVE_ERROR

# logistic_regression_forward_test_1()
# %% [markdown]
# # Generate Dataset
# %%
class DatasetGenerator:

    @staticmethod
    def generate_nonlinearly_separable(seed):
        x,y = make_moons(noise=0.15, random_state=seed)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor

    @staticmethod
    def generate_linearly_separable(seed):
        x, y = make_blobs(centers=2, random_state=seed, cluster_std=0.1)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor

    @staticmethod
    def generate_data(seed : int, func, ratio : float):
        points, classification = func(seed)
        x_train, x_test, y_train, y_test = train_test_split(points, classification, test_size = ratio)
        return [x_train, y_train], [x_test, y_test]
# %% [markdown]
# # Logistic Regression Analysis
# %%
class TrainRegression:
    def __init__(self):
        self.model = LogisticRegression()
        self.w = None

    @staticmethod
    def transformData(data):
        ones_tensor = torch.ones(data.shape[0], 1, device=device)
        return torch.cat((ones_tensor, data), dim=1)

    def train(self, training_data: list[torch.Tensor], iterations : int = 250, epsilon : float = 0, alpha : float = 0.1):
        data, labels = training_data
        data = self.transformData(data)
        dim = data.shape[1]
        self.w = torch.rand(dim, device=device).view(-1, 1)
        self.w, history_error = self.model.train(data, self.w , labels.view(-1, 1), iterations, alpha, epsilon)
        return self.w, history_error

    def medium_absolute_error(self, test_data: list[torch.Tensor]):
        data, labels = test_data
        return self.model.medium_absolute_error(self.transformData(data), self.w, labels.view(-1, 1))

    def accuracy(self, test_data: list[torch.Tensor]):
        data, labels = test_data
        return self.model.accuracy(self.transformData(data), self.w, labels.view(-1, 1))

    def make_line(self, m):
        y_line = -(self.w[0] + self.w[1] * m[:,0]) / self.w[2]
        return y_line
# %%
class Analysis:

    def __init__(self):
        self.iterations = 250
        self.test_partition = 0.3
        self.dataset_generator = DatasetGenerator()

    def make_table(self, table : Table, separable_errors : list[float], nonseparable_errors : list[float]):
        for i in range(self.iterations):
            table.add(i, "Linealmente separable", separable_errors[i])
            table.add(i, "No linealmente separable", nonseparable_errors[i])
            table.add(i, "Diferencia Absoluta", abs(separable_errors[i] - nonseparable_errors[i]))
            table.add(i, "Menor error", "Linealmente separable" if separable_errors[i] < nonseparable_errors[i] else "No linealmente separable")

    def analyze_data(self, seed : int, title : str, func):
        model = TrainRegression()
        training_data, test_data = self.dataset_generator.generate_data(seed, func, self.test_partition)
        w, errors = model.train(training_data, self.iterations)

        y_line = model.make_line(test_data[0])
        visualization.make_scatter_with_line(test_data[0][:,0], test_data[0][:,1], test_data[1], y_line, title)

        visualization.make_error_plot(errors,title)
        return errors

    def run(self, seed : int):
        error_table = Table(pd.DataFrame(),["Linealmente separable", "No linealmente separable", "Diferencia Absoluta"], "Diferencia_NONSE_SE")

        separable_errors = self.analyze_data(seed, "Linealmente separable", DatasetGenerator.generate_linearly_separable)
        nonseparable_errors = self.analyze_data(seed, "No linealmente separable", DatasetGenerator.generate_nonlinearly_separable)

        self.make_table(error_table, separable_errors, nonseparable_errors)

        error_table.show()
        error_table.generate_latex(False)

    def statistics(self, func, name : str):
        table = Table(pd.DataFrame(),["Promedio MAE(entrenamiento)", "Desviación estándar MAE(entrenamiento)", "MAE con datos de prueba"],name)
        error_list = torch.empty(0)
        std_list = torch.empty(0)

        for i in range(10):
            train_data, test_data = self.dataset_generator.generate_data(random.randint(0,10000), func, self.test_partition)
            model = TrainRegression()
            _, error = model.train(train_data, self.iterations)
            mae_mean = torch.tensor(error).mean()
            mae_std = torch.tensor(error).std()
            mae = model.medium_absolute_error(test_data)
            error_list = torch.cat((error_list, mae.unsqueeze(0)))
            std_list = torch.cat((std_list, mae_std.unsqueeze(0)))

            table.add(i, "Promedio MAE(entrenamiento)", mae_mean.item())
            table.add(i, "Desviación estándar MAE(entrenamiento)", mae_std.item())
            table.add(i, "MAE con datos de prueba", mae.item())
        table.show()
        table.generate_latex(True)
        return error_list.mean().item(), std_list.mean().item(), error_list.min().item()

    def run_many(self):
        separable_mean_error, separable_mean_std, separable_min_error = self.statistics(DatasetGenerator.generate_linearly_separable,"Linealmente_Separable")
        nonseparable_mean_error, nonseparable_mean_std, nonseparable_min_error = self.statistics(DatasetGenerator.generate_nonlinearly_separable,"No_Linealmente_Separable")

        compare_table = Table(pd.DataFrame(),[" ","Linealmente Separable", "No linealmente Separable"])
        compare_table.add(0," ","Promedio de MAE")
        compare_table.add(1," ","Promedio de Desviación estándar")
        compare_table.add(2," ","Mínimo error MAE obtenido")
        compare_table.add(0,"Linealmente Separable",separable_mean_error)
        compare_table.add(0,"No linealmente Separable",nonseparable_mean_error)
        compare_table.add(1,"Linealmente Separable",separable_mean_std)
        compare_table.add(1,"No linealmente Separable",nonseparable_mean_std)
        compare_table.add(2,"Linealmente Separable", separable_min_error)
        compare_table.add(2,"No linealmente Separable", nonseparable_min_error)
        compare_table.show()
        compare_table.generate_latex(False)

# %%
# analysis = Analysis()
# analysis.run(777)
# # %%
# analysis.run_many()
# %% [markdown]
# # Parte 2
# %% [markdown]
# ## Logistic Regression with TFIDF
# %%
nltk.download('stopwords')
stop_words_es = set(stopwords.words('spanish'))
# %% [markdown]
# ### Process Text
# %%
class TextProcessor:
    def __init__(self):
        self.train_vectorizer = TfidfVectorizer(analyzer=self.preprocess_document)
        self.cnt=0

    def tokenize(self, text : str):
        return re.findall(r"[a-záéíóúñü]+", text.lower())

    def preprocess_document(self, text : str):
        aux = [simplemma.lemmatize(token, lang='es') for token in self.tokenize(text) if token not in stop_words_es]
        self.cnt=max(self.cnt,len(aux))
        return aux

    def show_descriptor(self, train_vectorised, tokens):
        df = pd.DataFrame(train_vectorised.toarray(), columns = tokens)
        print(df.round(5))

    def make_descriptor(self, train_vectorised):
        return torch.tensor(train_vectorised.toarray(), dtype=torch.float32)

    def tfidf(self, corpus : list[str]):
        train_vectorized = self.train_vectorizer.fit_transform(corpus)
        tokens = self.train_vectorizer.get_feature_names_out()
        print(self.cnt, "tokens")
        return train_vectorized, tokens
# %%
CORPUS = [
    "¡Qué maravilloso es que nadie necesite esperar ni un solo momento antes de comenzar a mejorar el mundo!",
    "En un lugar de la Mancha, inmiscuyéndose en sustancias superfluas",
    "Nombras tú mi nombre Como jamás lo dijo un hombre Agapimú Tocas mi cintura Como la hiedra toca altura Agapimú",
    "La guerra es la paz. La libertad es la esclavitud. La ignorancia es la fuerza",
    "Tiemblas temblamos temblaremos temblaís tembló amor mío Como una gota de rocío Agapimú Entras en mi cuerpo Como la lluvia entra en mi huerto Agapimú",
    "Quien controla el pasado controla el futuro. Quien controla el presente controla el pasado",
]

def test_analyzer():
    text_processor = TextProcessor()
    for doc in CORPUS:
        print(text_processor.preprocess_document(doc))
    train_vectorized, tokens = text_processor.tfidf(CORPUS)
    text_processor.show_descriptor(train_vectorized, tokens)

# test_analyzer()
# %%
class ReadDataset:
    def __init__(self, path : str):
        self.path = path

    def read(self):
        dataset = pd.read_excel(self.path, sheet_name="Sheet1")
        self.corpus = []
        self.labels = []
        for i,row in dataset.iterrows():
            self.corpus.append(row["Segment"])
            self.corpus.append(row["Proposal"])
            self.labels.append(1)
            self.labels.append(0)

# read_dataset = ReadDataset("FEINA_1.xlsx")
# read_dataset.read()
# %%
def train_model(corpus, labels, test_size : float = 0.3):
    text_processor = TextProcessor()
    model = TrainRegression()

    data_vectorized, _ = text_processor.tfidf(corpus)
    data_matrix = text_processor.make_descriptor(data_vectorized)

    labels_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1).to(device)
    data_matrix = data_matrix.to(device)

    train_data, test_data, train_label, test_label = train_test_split(data_matrix, labels_tensor, test_size = test_size)

    model.train([train_data, train_label], 200, 1e-6, 1)

    mae = model.medium_absolute_error([test_data, test_label])
    acc = model.accuracy([test_data, test_label])

    print("MAE: ", mae.item())
    print("Accuracy: ", acc.item())

    return mae, acc

# train_model(read_dataset.corpus, read_dataset.labels)
# %% [markdown]
# # Logistic Regression with Embeddings Vectors
# %%
def BERT_train(corpus, labels ,test_size : float = 0.3):
    model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")
    model.to(device)

    embeddings = model.encode(corpus, convert_to_tensor=True, normalize_embeddings=True)
    train_data, test_data, train_label, test_label = train_test_split(embeddings, labels, test_size = test_size)

    lr_model = TrainRegression()
    lr_model.train([train_data, torch.tensor(train_label).to(device)], 200, 1e-6, 1)

    mae = lr_model.medium_absolute_error([test_data, torch.tensor(test_label).to(device)])
    acc = lr_model.accuracy([test_data, torch.tensor(test_label).to(device)])
    print("Medium Absolute Error", mae.item())
    print("Accuracy", acc.item())

    return mae,acc

# BERT_Train(read_dataset.corpus, read_dataset.labels)