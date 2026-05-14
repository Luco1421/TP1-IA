# %% [markdown]
# # Start of project
# %%
import sys, os

os.system(f'{sys.executable} -m pip install pip --upgrade')
os.system(f'{sys.executable} -m pip install -U torch --index-url https://download.pytorch.org/whl/cu128')
os.system(f'{sys.executable} -m pip install -U transformers pandas')
# %% [markdown]
# # Imports
# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import pandas as pd
import json
import random
from TP2 import train_model, BERT_train, ReadDataset, Table

import os
# %% [markdown]
# # Enviromental Variables
# %%
os.environ["HF_TOKEN"] = "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"
MAX_SIZE = 4000
# %% [markdown]
# # Dataset Utils
# %% [markdown]
# # Prompts
# %%
CONTEXT = """
Eres un clasificador de textos financieros, que clasifica cada texto con: 0 si es simple y 1 si es complejo.
Lee los textos a continuación y responde solo en JSON con los números de 'ID' y 'CLASE'.

"""

TEXT_BEGINER = """
Textos a clasificar:

"""

#Texto complejo
SAMPLE_1 = "1. \"El varias veces mencionado Yager, nos dice acertadamente sobre estos aspectos lo siguiente: La mayoría de ustedes están hambrientos de tener libertad financiera, están hastiados de tener batallas financieras porque son como un carrusel.\" \n"

#Texto simple
SAMPLE_2 = "2. \"El equipo realizó la redacción del libro luego de obtener la información anterior y reunir las fuentes bibliográficas y virtuales respectivas.\" \n"

#Texto simple
SAMPLE_3 = "3. \"La mayoría de los casos involucra la implementación de estrategias de inversión muy especializadas desde el inicio.\" \n"

SHOTS_BEGINER = """
Por ejemplo lo siguientes textos se clasifican como:

"""

def extract_results(content : str):
    content = json.loads(content[content.find('[') : content.rfind(']') + 1])
    answer = [None] * len(content)
    for dic in content:
        answer[int(dic['ID']) - 1] = int(dic['CLASE'])
    return answer
# %% [markdown]
# # Llama Model
# %% [markdown]
# ## Charge model
# %%
model_name = "HuggingFaceTB/SmolLM3-3B"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
).to(device)
# %% [markdown]
# ## Function to request llama's answer
# %%
import time
import torch

def chat_hf_tb(text: str, prompt: str):
    t0 = time.perf_counter()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    t1 = time.perf_counter()

    model_inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True
    ).to(model.device)
    t2 = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
    t3 = time.perf_counter()

    output_ids = generated_ids[0, model_inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    t4 = time.perf_counter()

    print(f"template:  {t1 - t0:.4f}s")
    print(f"tokenize:  {t2 - t1:.4f}s")
    print(f"generate:  {t3 - t2:.4f}s")
    print(f"decode:    {t4 - t3:.4f}s")
    print(f"in_tokens: {model_inputs.input_ids.shape[1]}")
    print(f"out_tokens:{output_ids.shape[0]}")

    return extract_results(response_text)
# %% [markdown]
# ### Example of use
# %%
chat_hf_tb(SAMPLE_1 + SAMPLE_2 + SAMPLE_3, CONTEXT + TEXT_BEGINER)
# %% [markdown]
# # Stadistics of Results
# %%
class TestModels :
    def __init__(self, path : str):
        self.read_dataset = ReadDataset(path)
        self.read_dataset.read()
        self.corpus, self.labels = self.read_dataset.corpus, self.read_dataset.labels
        self.indexes = [i for i in range(len(self.corpus))]

    def rearrenge(self):
        random.shuffle(self.indexes)

    def set_batches(self, index : list[int]):
        batches = []
        batch = ""
        id = 1
        for i in index:
            batch += f"{id}. \"{self.corpus[i]}\" \n"
            id+=1
            if len(batch) > MAX_SIZE :
                batches.append(batch)
                batch = ""
                id = 1
        if len(batch) > 0 :
            batches.append(batch)
        return batches

    def set_shots(self, index : list[int]):
        batch = ""
        id = 1
        for i in index:
            batch += f"{id}. \"{self.corpus[i]}\" \n Clasificación: {self.labels[i]} \n"
            id+=1
        return batch

    def test(self, func, index : list[int], name : str, shots : str = ""):
        batches = self.set_batches(index)
        position = 0
        cnt = 0
        prompt = CONTEXT
        if len(shots) > 0:
            prompt += SHOTS_BEGINER + shots
        prompt += TEXT_BEGINER
        for batch in batches:
            for l in func(batch, prompt):
                cnt += self.labels[index[position]] == l
                position += 1
        return cnt / len(index)

    def run_without_shots(self, func, total_samples, name : str):
        self.test(func, self.indexes[: total_samples], name)

    def run_with_shots(self, func, k : int, name : str):
        shots = self.set_shots(self.indexes[-k: ])
        self.test(func, self.indexes, name, shots)

test_models = TestModels("FEINA_1.xlsx")
# %% [markdown]
# # Test without Few Shots
# %%
test_models.run_without_shots(chat_hf_tb,10,"Hugging Face TB")
# test_models.run_without_shots()
# %% [markdown]
# # Test with Few Shots
# %%
def test_llm_with_shots():
    for i in range(2,5):
        test_models.rearrenge()
    #     test_models.run_with_shots(, i)
    #     test_models.run_with_shots(, i)

# %% [markdown]
# 
# %%

def test_all_LLM():
    table = Table(pd.MultiIndex.from_product([
        ["Regresión Lineal","BERT + Regresión Lineal","Hugging Face Tb","Phi 4 mini"],
        ["Desviación Estándar MAE","Media MAE"]
    ]),"Tabla30Iter")
    test_models.rearrenge()
    test_models.run_without_shots(chat_hf_tb, 100, "HF_LLM")
    test_models.run_without_shots(chat_qwen, 100, "Qwen_LLM")
    for i in range(30):
        test_models.rearrenge()
        # test_models.run_without_shots(model1, 100)
        # test_models.run_without_shots(model2, 100)
        # for k in range(2, 5):
        #     test_models.run_with_shots(model1, k)
        #     test_models.run_with_shots(model2, k)
