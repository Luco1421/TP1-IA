# Figuras de la documentación

Exporta aquí los gráficos generados por `TP3.ipynb` (usa `plt.savefig(...)` o el
botón de guardar de cada figura). El documento `main.tex` los incluye
automáticamente si existen; si falta alguno, en su lugar aparece un recuadro con
el nombre del archivo esperado, de modo que el PDF compila igual.

Nombres esperados:

| Archivo                     | Origen en `TP3.ipynb`                                            |
|-----------------------------|-----------------------------------------------------------------|
| `xor_arquitectura.png`      | Diagrama de la red XOR (2-2-2), hecho a mano / herramienta       |
| `xor_frontera.png`          | `make_decision_boundary_plot` en `objective_xor`                 |
| `xor_error.png`             | `make_error_plot` en `objective_xor`                             |
| `r2_separable.png`          | `make_scatter` (datos `generate_linearly_separable`)             |
| `r2_noseparable.png`        | `make_scatter` (datos `generate_nonlinearly_separable`)          |
| `r2_curvas_2x2.png`         | grilla 2x2 de `run_average_curves` / `run_50_iterations`         |
| `acrima_mlp_error.png`      | `make_error_plot` en `obtain_best_parameters_acrima`             |
| `final_mlp_curva.png`       | `make_error_plot` "Entrenamiento MLP" en `run_many`              |
| `final_alex_curva.png`      | `make_error_plot` "Entrenamiento Alex" en `run_many`             |
