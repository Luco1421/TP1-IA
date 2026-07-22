"""! @brief Utils for testing LLM's"""
# @file LLMs.py
#
# @mainpage LLMs
#
# @section description_main Description
# This module provides a collection of utilities, functions, and
# evaluation pipelines designed to test and compare the performance
# of Large Language Models (LLMs) under different prompting strategies,
# including zero-shot and few-shot settings.
#
# @section libraries_main Libraries
# - torch (https://docs.pytorch.org/docs/stable/index.html)
#   - Access to tensor and all its operations
# - pandas (https://pandas.pydata.org/docs/)
#   - Used to generate tables
# - random (https://docs.python.org/3/library/random.html)
#   - Used to shuffle the dataset
# - re (https://docs.python.org/es/3/library/re.html)
#   - Used to create regular expressions
# - os (https://docs.python.org/es/3.10/library/os.html)
#   - Used to install dependencies
# - transformers (https://pypi.org/project/transformers/)
#   - Used to load the model
#
# @section authors_main Authors
# - Alejandro Cerdas
# - Kener Castillo
# - Pablo Perez




# @brief Imports and Global Variables defintions
# Imports
from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import random
import torch
import re
import os

# HuggingFace authentication token — set HF_TOKEN in your environment before running
HF_TOKEN = os.getenv("HF_TOKEN")
TOKEN = HF_TOKEN if HF_TOKEN else None

# Select computation device (GPU if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimize CUDA memory allocation behavior
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Ratio used for train/test splitting
RATIO_SPLIT_4C = 0.2

# Maximum sequence length for model inference
MAX_SIZE = 4096

# Prompt defining the classification task
CONTEXT = """
Eres un clasificador de textos financieros.

Clasifica cada texto con las siguientes clases:

0 = simple
1 = complejo

Responde únicamente una línea por texto, en este formato exacto:

ID:CLASE

Ejemplo de respuesta válida:

1:0

Donde 1 es el ID y 0 es la CLASE.

"""

# Prompt section introducing input texts
TEXT_BEGINNER = """
Textos a clasificar:
"""

# Prompt section introducing few-shot examples
SHOTS_BEGINNER = """
Ejemplos:
"""

# Complex text sample
SAMPLE_1 = "1. \"El varias veces mencionado Yager, nos dice acertadamente sobre estos aspectos lo siguiente: La mayoría de ustedes están hambrientos de tener libertad financiera, están hastiados de tener batallas financieras porque son como un carrusel.\" \n"

# Simples texts samples
SAMPLE_2 = "2. \"El equipo realizó la redacción del libro luego de obtener la información anterior y reunir las fuentes bibliográficas y virtuales respectivas.\" \n"
SAMPLE_3 = "3. \"La mayoría de los casos involucra la implementación de estrategias de inversión muy especializadas desde el inicio.\" \n"




# @brief Dataset reading utilities for text classification tasks.

class ReadDataset:
    """! Utility class for reading datasets from Excel files.

    Loads text segments and proposals from a spreadsheet
    and creates a labeled corpus for binary classification.
    """

    def __init__(self, path : str):
        """! Initializes the dataset reader.

        @param path Path to the Excel dataset file.
        """

        # Path to the dataset file
        self.path = path

        # List used to store text documents and labels
        self.corpus = []
        self.labels = []


    def read(self):
        """! Reads and processes the dataset.

        Loads the Excel file, extracts text samples,
        and assigns binary labels to the corpus.
        """

        # Read Excel dataset from Sheet1
        dataset = pd.read_excel(self.path, sheet_name="Sheet1")

        # Iterate through dataset rows
        for i,row in dataset.iterrows():

            # Store text in corpus
            self.corpus.append(row["Segment"])
            self.corpus.append(row["Proposal"])

            # Store lables
            self.labels.append(1)
            self.labels.append(0)




# @brief Utility class for batching text data.
class Batcher:
    """! Utility class for batching text data.

    Create batches of text documents for model inference.
    Create batches of text documents and labels for few-shot evaluation.
    """

    def __init__(self):
        """! Initializes the batcher.
        """
        # Maximum sequence length for model inference
        self.max_size = MAX_SIZE

    def set_batches(self, corpus: list[str]):
        """! Splits a list of text documents into batches.

        @param corpus List of text documents.
        @return List of batches of text documents.
        """
        # result
        batches = []
        # save current batch
        batch = ""
        # id of the current text in the batch
        id = 1
        for doc in corpus:
            # add the text to the current batch
            batch += f"{id}. \"{doc}\" \n"
            # increment the id
            id += 1
            # save and reset the batch if exceeds the max size
            if len(batch) > self.max_size:
                batches.append(batch)
                batch = ""
                id = 1
        # save the last batch if it is not empty
        if len(batch) > 0:
            batches.append(batch)
        return batches

    def set_shots(self, corpus: list[str], labels: list[int]):
        """! Create a string of documents with their labels to use as few-shot

        @param corpus List of examples text
        @param List of labels for the examples

        @return String few-shots examples
        """
        # result
        batch = ""

        # id of the current text in the batch
        id = 1

        for i in range(len(corpus)):

            # append text with id and label
            batch += f"{id}. \"{corpus[i]}\" \n Clasificación: {labels[i]} \n"
            id += 1

        return batch




# @brief Utility class for evaluating Large Language Models (LLMs).

class TestLLMUtils :
    """! Evaluation utilities for LLM benchmarking.

    Supports zero-shot, few-shot, and multi-run evaluation
    with statistical aggregation of results.
    """

    def __init__(self):
        """! Initializes the evaluation utility.
        """

        # Batch utility for handling corpus splitting
        self.batcher = Batcher()

    def test(self, func, batches : list[str], labels : list[int], shots : str = ""):
        """! Computes accuracy for a given LLM function.

        @param func Function that queries the LLM.
        @param batches Input text batches.
        @param labels Ground-truth labels.
        @param shots Optional few-shot prompt examples.
        @return Accuracy score.
        """

        # Current prediction index
        position = 0

        # Correct predictions counter
        cnt = 0

        # Build system prompt
        prompt = CONTEXT

        # Add few-shot examples if provided
        if len(shots) > 0:
            prompt += SHOTS_BEGINNER + shots

        # Append input text section
        prompt += TEXT_BEGINNER

        # Iterate through batches
        for batch in batches:

            # Get model predictions for batch
            for l in func(batch, prompt):

                # Compare prediction with ground truth
                cnt += labels[position] == l
                position += 1

        # Return accuracy
        return cnt / position

    def test_without_shots(self, name, fun_model, dataset):
        """! Evaluates LLM in zero-shot setting.

        @param name Model name.
        @param fun_model LLM inference function.
        @param dataset Dataset containing corpus and labels.
        """

        # Split dataset into batches
        batches = self.batcher.set_batches(dataset.corpus)

        # Extract labels
        labels = dataset.labels

        # Compute accuracy without few-shot examples
        accuracy = self.test(fun_model, batches, labels)

        # Display result
        print(f"{name} - Accuracy: {accuracy}")

    def test_with_shots(self, name, fun_model, dataset, conf_shots):
        """! Evaluates LLM using multiple few-shot configurations.

        @param name Model name.
        @param fun_model LLM inference function.
        @param dataset Dataset containing corpus and labels.
        @param conf_shots List of few-shot configurations.
        """

        # Determine number of reserved examples for shots
        reserved_shots = max(conf_shots)

        # Create test batches excluding shot samples
        batches = self.batcher.set_batches(dataset.corpus[reserved_shots:])

        # Corresponding labels for test set
        labels = dataset.labels[reserved_shots:]

        # Evaluate each shot configuration
        for i in conf_shots:

            # Generate few-shot prompt examples
            shots = self.batcher.set_shots(dataset.corpus[:i], dataset.labels[:i])

            # Compute accuracy with few-shot prompting
            accuracy = self.test(fun_model, batches, labels, shots)

            # Display result
            print(f"{name} with {i} shots - Accuracy: {accuracy}")

    def test_all_LLM(self, name, fun_model, dataset, shot_configs, n_runs=30):
        """! Runs statistical evaluation for zero-shot and few-shot LLM settings.

        @details
        Executes multiple randomized experiments, computes accuracy
        distributions for different few-shot configurations, and
        reports mean and standard deviation.

        @param name Model name.
        @param fun_model LLM inference function.
        @param dataset Dataset containing corpus and labels.
        @param shot_configs List of few-shot configurations.
        @param n_runs Number of random evaluation runs.
        @return Dictionary with accuracy results per configuration.
        """

        # Number of training samples
        N = int(len(dataset.corpus) * (1 - RATIO_SPLIT_4C))

        # Dictionary to store results
        results = {}

        # Initialize zero-shot results list
        results[0] = []

        # Initialize few-shot results lists
        for shots in shot_configs:
            results[shots] = []

        # Repeat evaluation over multiple runs
        for i in range(n_runs):

            # Generate shuffled dataset indices
            indexes = list(range(len(dataset.corpus)))

            # Deterministic random generator
            rng = random.Random(i)

            # Shuffle dataset
            rng.shuffle(indexes)

            # Reorder corpus and labels
            shuffle_corpus = [dataset.corpus[idx] for idx in indexes]
            shuffle_labels = [dataset.labels[idx] for idx in indexes]

            # Create test batches
            test_batches = self.batcher.set_batches(shuffle_corpus[N:])

            # Evaluate accuracy
            acc_zero = self.test(fun_model, test_batches, shuffle_labels[N:], shots="")
            results[0].append(acc_zero)

            # Evaluate configurations
            for shots_count in shot_configs:

                # Build prompt
                shots = self.batcher.set_shots(shuffle_corpus[:shots_count],shuffle_labels[:shots_count])

                #Evaluate model accuracy
                acc_i = self.test(fun_model, test_batches, shuffle_labels[N:], shots)

                # Store result
                results[shots_count].append(acc_i)

        # Convert zero-shot results to tensor
        acc_tensor = torch.tensor(results[0])

        # Print without shots statistics
        print(f"{name} without shots: average = {acc_tensor.mean().item():.4f}, std = {acc_tensor.std().item():.4f}")

        # Print few-shot statistics
        for shots_count, acc_list in results.items():
            if shots_count == 0:
                continue
            acc_tensor = torch.tensor(acc_list)
            print(f"{name} with {shots_count} shots - average = {acc_tensor.mean().item():.4f}, std = {acc_tensor.std().item():.4f}")

        # Separator line
        print("-" * 50)

        # Return full results dictionary
        return results




# @brief Utility functions for model inference, loading, and GPU cleanup.
def extract_results(content: str):
    """! Extracts classification results from LLM output.

    @details
    Parses model output lines in the format `ID:CLASE`
    and converts them into a list of integer predictions.
    Invalid lines are mapped to -1.

    @param content Raw text generated by the model.
    @return List of predicted class labels.
    """

    # List for save answers
    answers = []

    # Loop for interate in lines of content
    for raw_line in content.splitlines():

        # Use regular expression to extract the ID and the label
        m = re.match(r"^(\d+):([01])$", raw_line)

        # If answer is not valid, save -1
        answers.append(int(m.group(2)) if m else -1)

    # Return answers list
    return answers

def charge_model(model_name: str):
    """! Loads a HuggingFace causal language model and tokenizer.

    @param model_name Name of the pretrained model.
    @return Tuple (tokenizer, model).
    """

    # Charge tokenizer of model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token = TOKEN # Token for HuggingFace
    )

    # Charge weights of model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token = TOKEN, # Token for HuggingFace
        dtype = torch.bfloat16 # should be torch.float32
    ).to(DEVICE)

    # Return tokenizer and model
    return tokenizer, model

def clean_gpu():
    """! Clears GPU memory cache.
    """

    # Clear cuda
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
