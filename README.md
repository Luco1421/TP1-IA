# TPs IA

Three assignments ("Trabajos Prácticos") for the *Inteligencia Artificial*
course at Tecnológico de Costa Rica, done as a 3-person team (Alejandro
Cerdas, Kener Castillo, Pablo Pérez). Each folder is self-contained, with
its own notebooks, data, and a LaTeX report under `documentation/` /
`documentacion/`.

- **[TP1](TP1/)** — Gradient Descent, RMSProp, and CMA-ES implemented from
  scratch in PyTorch, optimizing three 2D benchmark functions with
  Optuna-tuned hyperparameters.
- **[TP2](TP2/)** — Comparing TF-IDF, sentence embeddings, and prompted LLMs
  (Phi-4, SmolLM3) for classifying Spanish financial text as simple vs.
  complex, with Friedman/Wilcoxon statistical significance testing across
  the methods.
- **[TP3](TP3/)** — A multilayer perceptron built from scratch (backprop,
  momentum, Xavier init) benchmarked on XOR and synthetic data, then
  compared against transfer-learned AlexNet on a real glaucoma
  fundus-image dataset.

No unified dependency file exists across the three — each notebook installs
or documents its own requirements; see each TP's README for specifics.
