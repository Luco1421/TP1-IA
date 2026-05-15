from TP2 import train_model, BERT_train, Batcher

from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import random
import torch
import re
import os
import gc

os.environ["HF_TOKEN"] = "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" # Change
HF_TOKEN = os.getenv("HF_TOKEN")
TOKEN = HF_TOKEN if HF_TOKEN else None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def extract_results(content: str):
    answers = []
    for raw_line in content.splitlines():
        m = re.match(r"^(\d+):([01])$", raw_line)
        answers.append(int(m.group(2)) if m else -1)
    return answers

def charge_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token = TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token = TOKEN,
        dtype = torch.bfloat16 # should be torch.float32
    ).to(DEVICE)

    return tokenizer, model

def clean_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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

TEXT_BEGINNER = """
Textos a clasificar:
"""

SHOTS_BEGINNER = """
Ejemplos:
"""

#Texto complejo
SAMPLE_1 = "1. \"El varias veces mencionado Yager, nos dice acertadamente sobre estos aspectos lo siguiente: La mayoría de ustedes están hambrientos de tener libertad financiera, están hastiados de tener batallas financieras porque son como un carrusel.\" \n"

#Texto simple
SAMPLE_2 = "2. \"El equipo realizó la redacción del libro luego de obtener la información anterior y reunir las fuentes bibliográficas y virtuales respectivas.\" \n"

#Texto simple
SAMPLE_3 = "3. \"La mayoría de los casos involucra la implementación de estrategias de inversión muy especializadas desde el inicio.\" \n"

class TestLLMUtils :
    def __init__(self):
        self.batcher = Batcher()

    def test(self, func, batches : list[str], labels : list[int], shots : str = ""):
        position = 0
        cnt = 0
        prompt = CONTEXT
        if len(shots) > 0:
            prompt += SHOTS_BEGINNER + shots
        prompt += TEXT_BEGINNER
        for batch in batches:
            for l in func(batch, prompt):
                cnt += labels[position] == l
                position += 1
        return cnt / position

    def test_without_shots(self, name, fun_model, dataset):
        batches = self.batcher.set_batches(dataset.corpus)
        labels = dataset.labels
        accuracy = self.test(fun_model, batches, labels)
        print(f"{name} - Accuracy: {accuracy}")

    def test_with_shots(self, name, fun_model, dataset, conf_shots):
        reserved_shots = max(conf_shots)
        batches = self.batcher.set_batches(dataset.corpus[reserved_shots:])
        labels = dataset.labels[reserved_shots:]
        for i in conf_shots:
            shots = self.batcher.set_shots(dataset.corpus[:i], dataset.labels[:i])
            accuracy = self.test(fun_model, batches, labels, shots)
            print(f"{name} with {i} shots - Accuracy: {accuracy}")
