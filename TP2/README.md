# TP2 — Classical ML vs. embeddings vs. LLMs for text-complexity classification

Binary classification of Spanish financial-education text segments as
*simple* vs. *complex*, using the FEINA dataset (`FEINA_1.xlsx`, ~10,600
labeled complex/simplified text pairs). Four approaches are implemented and
compared:

1. **Perceptron and logistic regression from scratch**
   (`LogisticRegression.ipynb`) — hand-written forward pass, gradients, and
   training loop in PyTorch tensors (no `nn.Module`), with an early-stopping
   criterion based on gradient-norm convergence, validated on synthetic
   linearly- and non-linearly-separable data before use on the real task.
2. **TF-IDF + logistic regression** (`compareTFIDF.py`) — sweeps 9
   tokenizer × lemmatizer combinations (NLTK, regex, spaCy, Stanza tokenizers;
   WordNet/NLTK, Simplemma, spaCy, Stanza lemmatizers), each evaluated over
   3 seeds with an Optuna-calibrated learning rate. Best pipeline: Stanza
   tokenizer + NLTK stopwords + Stanza lemmatizer.
3. **Sentence embeddings + logistic regression** (`compareEmbeddings.py`) —
   compares `distiluse-base-multilingual-cased-v1`,
   multilingual BERT, `dccuchile/bert-base-spanish-wwm-cased` (BETO), and
   `Buzzeitor/roberta-base-bne`.
4. **Prompted LLMs** (`LLMs.py`, `Phi4_LLM.ipynb`, `SmolLM3.ipynb`) —
   Microsoft **Phi-4-mini-instruct** and **SmolLM3**, run locally
   (bfloat16, temperature/top_p = 0.1) under zero-shot and few-shot
   (2/4/7-shot) prompting, parsed from a structured `ID:CLASE` output.

`StadisticTest.ipynb` compares all methods statistically: 30 paired 80/20
runs across 6 treatments, analyzed with a **Friedman test** for global
difference, **Kendall's W** for effect size, and pairwise **Wilcoxon
signed-rank tests with Holm correction**.

## Results

Across 30 runs: TF-IDF ≈48.0% precision, embeddings ≈47.9%, SmolLM3
≈49.8%, Phi-4 ≈49.8% — all close to chance level on this balanced binary
task. The Friedman test finds the treatments statistically different
overall (χ²=72.23, p≈3.5×10⁻¹⁴), and the LLM-based methods differ
significantly from the logistic-regression-based ones (p<0.001), but no
method reaches practically useful accuracy — the full analysis and
methodology are in `documentation/docu.pdf`.

## Stack

PyTorch, scikit-learn, sentence-transformers, Hugging Face `transformers`,
NLTK, spaCy (`es_core_news_lg`), Stanza (Spanish models bundled in
`stanza_models/`), Simplemma, Optuna, SciPy, statsmodels.

## Running

`LogisticRegression.ipynb`, `compareTFIDF.py`, and `compareEmbeddings.py`
run on CPU. `Phi4_LLM.ipynb` and `SmolLM3.ipynb` need a CUDA GPU in
practice (tested on an RTX 3060 8GB; a full run takes 26–86 minutes per
configuration) and a Hugging Face token exported as the `HF_TOKEN`
environment variable before running `LLMs.py`.
