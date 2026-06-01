def run_average_curves(self, num_runs: int = 50, num_epochs: int = 1000,
                       ratio: float = 0.2, alpha: float = 0.01):
    n_cfg = len(self.config)
    # Acumuladores en GPU: nada de .item() dentro del loop
    train_curves = [torch.zeros(num_epochs, dtype=TYPE, device=DEVICE) for _ in range(n_cfg)]
    val_curves = [torch.zeros(num_epochs, dtype=TYPE, device=DEVICE) for _ in range(n_cfg)]
    final_val = [torch.zeros(num_runs, dtype=TYPE, device=DEVICE) for _ in range(n_cfg)]

    for run in range(num_runs):
        train_data, val_data = DatasetGenerator.split_data(ratio, run, self.data,
                                                           self.labels)
        Xtr, Ttr = train_data[0], train_data[1].view(-1, 1)
        Xva, Tva = val_data[0], val_data[1].view(-1, 1)

        for j, (M, gamma) in enumerate(self.config):
            mlp = MultilayerPerceptron([2, M, 1], alpha, gamma, 1e5)
            for ep in range(num_epochs):
                mlp.forward(Xtr)
                train_curves[j][ep] += mse_loss(mlp.Ys, Ttr)  # tensor 0-dim, sin sync
                mlp.backpropagate_deltas(Ttr)
                mlp.update_weights(Xtr)

                mlp.forward(Xva)
                val_curves[j][ep] += mse_loss(mlp.Ys, Tva)

            final_val[j][run] = mse_loss(mlp.Ys, Tva)

    # Promedio y un único traslado a CPU al final
    train_np = [(c / num_runs).cpu().numpy() for c in train_curves]
    val_np = [(c / num_runs).cpu().numpy() for c in val_curves]
    final_mean = [final_val[j].mean().item() for j in range(n_cfg)]
    final_std = [final_val[j].std().item() for j in range(n_cfg)]

    # --- Tabla de gráficas iteraciones vs error (grilla 2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for j, (M, gamma) in enumerate(self.config):
        ax = axes[j // 2, j % 2]
        ax.plot(train_np[j], label="Entrenamiento")
        ax.plot(val_np[j], label="Validación")
        ax.set_title(f"M = {M}   γ = {gamma}")
        ax.set_xlabel("Iteración");
        ax.set_ylabel("Error MSE");
        ax.legend()
    fig.suptitle(f"Curva promedio sobre {num_runs} corridas (80/20), α = {alpha}")
    fig.tight_layout()
    plt.show()

    # --- Tabla de error final de convergencia ---
    table = Table(pd.DataFrame(),
                  ["M", "Gamma", "Error final (val)", "Desv. est."],
                  "Comparación_Configuraciones")
    for j, (M, gamma) in enumerate(self.config):
        ind = table.obtain_row_count()
        table.add(ind, "M", M)
        table.add(ind, "Gamma", gamma)
        table.add(ind, "Error final (val)", final_mean[j])
        table.add(ind, "Desv. est.", final_std[j])
    table.show()
    table.latex()