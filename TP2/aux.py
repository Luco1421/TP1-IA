#!pip install stanza spacy
import re
import pandas as pd
import simplemma
import torch

from sklearn.feature_extraction.text import TfidfVectorizer


#  Probar con stanza y spacy para comparar
# Cambiar a nltk tokenize en lugar de re, aunque no se si es lo mismo o mejoraria algo, igual es biblioteca menos
#Se puede usar el TfidfVectorizer para tokenizar?

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

########################################################################################################################

#!pip install transformers

# Actualmente con un modelo multilenguaje, pero probar con esto a ver si mejora

from transformers import BertTokenizer, BertModel

modelo = 'dccuchile/bert-base-spanish-wwm-cased'

tokenizer = BertTokenizer.from_pretrained(modelo)
bert = BertModel.from_pretrained(modelo)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo.to(device)

corpus = [
    "Hola, estoy feliz",
    "Hola, estoy triste",
    "Hola, estoy feliz y triste",
    "Hola, estoy hecho caca"
]

inputs = tokenizer( corpus,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                    )

with torch.no_grad():
  outputs = bert(**inputs)

embeddings = outputs.pooler_output


### Hablar de SUGEVAL y esas cosas, ademas investigar el numero de iteraciones y el alpha, comparar bastante con tablas
## Hablar de los prompts probados