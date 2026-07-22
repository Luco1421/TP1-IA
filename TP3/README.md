# TP3 — Multilayer perceptron from scratch vs. transfer-learned AlexNet

A 3-layer multilayer perceptron implemented entirely from scratch
(`TP3.ipynb`): manual forward pass, backpropagated deltas, gradient descent
with momentum, and Xavier/Glorot initialization, validated against
hand-computed expected values with numeric unit tests. It's benchmarked on
three problems of increasing difficulty:

1. **XOR** — hyperparameters tuned with Optuna over a 20,000-epoch budget.
2. **Synthetic 2D data** (`make_blobs`, linearly separable, and
   `make_moons`, non-linearly separable) — 4 hidden-size/momentum
   configurations, averaged over 50 random 80/20 splits.
3. **ACRIMA glaucoma fundus-image classification** (705 real images: 396
   glaucoma, 309 normal) — the MLP is calibrated at three input resolutions
   (32/48/64 px) via Optuna, then compared against **AlexNet with
   ImageNet-pretrained weights** (`torchvision.models.alexnet`, output
   layer swapped for binary classification), fine-tuned with SGD +
   momentum.

## Results

Over 10 random 80/20 splits on the ACRIMA task: transfer-learned AlexNet
reached **99.36% mean test accuracy** (σ=0.87%, 7/10 runs at 100%), against
**84.29%** (σ=4.86%) for the best from-scratch MLP configuration
(64 px input, 118 hidden units). The gap is attributed to AlexNet's
convolutional weight-sharing and ImageNet transfer learning compensating
for the small training set (~564 images) — full methodology and figures in
`documentacion/main.pdf`.

`alexnet_demo.ipynb` is a separate warm-up exercise (not part of the ACRIMA
comparison): fine-tunes pretrained AlexNet on MNIST for 5 epochs.

## Stack

PyTorch, torchvision (`models.alexnet`, `ImageFolder`), Optuna,
scikit-learn (`make_blobs`, `make_moons`), pandas, matplotlib, Pillow.

## Running

Open `TP3.ipynb` in Jupyter; expects the `dataset/` folder (ACRIMA images,
included) laid out for `torchvision.datasets.ImageFolder`. AlexNet
fine-tuning benefits strongly from a CUDA GPU, though it will run (slowly)
on CPU. `alexnet_demo.ipynb` auto-downloads MNIST if not already present
under `data/`.
