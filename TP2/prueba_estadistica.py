from itertools import combinations

import torch
from scipy.stats import chi2, wilcoxon


# Datos de las 30 corridas. Cada posicion i representa la misma corrida/semilla
# para todos los tratamientos.
BERT = torch.tensor([
    0.4807, 0.4799, 0.4824, 0.4774, 0.4849, 0.4766, 0.4783, 0.4770,
    0.4783, 0.4788, 0.4788, 0.4801, 0.4782, 0.4749, 0.4780, 0.4797,
    0.4824, 0.4745, 0.4789, 0.4802, 0.4810, 0.4758, 0.4824, 0.4785,
    0.4784, 0.4807, 0.4779, 0.4774, 0.4747, 0.4764,
], dtype=torch.float64)

TFIDF = torch.tensor([
    0.4876, 0.4803, 0.4832, 0.4792, 0.4794, 0.4805, 0.4834, 0.4852,
    0.4781, 0.4859, 0.4753, 0.4791, 0.4827, 0.4795, 0.4788, 0.4812,
    0.4814, 0.4819, 0.4760, 0.4748, 0.4781, 0.4782, 0.4795, 0.4761,
    0.4761, 0.4799, 0.4796, 0.4817, 0.4816, 0.4810,
], dtype=torch.float64)

SMOLLM3_SIN_SHOTS = torch.tensor([
    0.488905325443787, 0.5150602409638554, 0.48628048780487804,
    0.4809451219512195, 0.5129573170731707, 0.4880774962742176,
    0.5069801616458487, 0.4958615500376223, 0.50187265917603,
    0.5143072289156626, 0.5038402457757296, 0.48153730218538054,
    0.5195783132530121, 0.5123226288274833, 0.4954819277108434,
    0.4771341463414634, 0.49776119402985075, 0.5193452380952381,
    0.48134044173648133, 0.4713855421686747, 0.49255952380952384,
    0.5003717472118959, 0.5, 0.5191873589164786, 0.4915514592933948,
    0.5163249810174639, 0.4992378048780488, 0.49276466108149275,
    0.48698884758364314, 0.4784905660377359,
], dtype=torch.float64)

SMOLLM3_2_SHOTS = torch.tensor([
    0.4948224852071006, 0.49623493975903615, 0.5022865853658537,
    0.504950495049505, 0.4969512195121951, 0.5216095380029806,
    0.4996326230712711, 0.510158013544018, 0.49887640449438203,
    0.5015060240963856, 0.5084485407066052, 0.4777694046721929,
    0.4894578313253012, 0.5093353248693054, 0.49849397590361444,
    0.4634146341463415, 0.4847356664184661, 0.5133928571428571,
    0.48514851485148514, 0.4932228915662651, 0.4880952380952381,
    0.5200892857142857, 0.5038109756097561, 0.5214446952595937,
    0.4869431643625192, 0.5195488721804511, 0.4878048780487805,
    0.49047981721249045, 0.47505584512285925, 0.5049056603773585,
], dtype=torch.float64)

PHI4_SIN_SHOTS = torch.tensor([
    0.4822485207100592, 0.5075301204819277, 0.4923780487804878,
    0.5060975609756098, 0.5121951219512195, 0.503725782414307,
    0.4915378955114054, 0.5026335590669676, 0.4868913857677903,
    0.526355421686747, 0.5092165898617511, 0.4860587792012057,
    0.5060240963855421, 0.48618371919342795, 0.516566265060241,
    0.5060975609756098, 0.4873134328358209, 0.4955357142857143,
    0.5201520912547528, 0.4977409638554217, 0.47023809523809523,
    0.46651785714285715, 0.506859756097561, 0.5063957863054929,
    0.49385560675883255, 0.49097744360902257, 0.4946646341463415,
    0.5072353389185073, 0.47505584512285925, 0.4950943396226415,
], dtype=torch.float64)

PHI4_2_SHOTS = torch.tensor([
    0.48298816568047337, 0.509789156626506, 0.48628048780487804,
    0.5053353658536586, 0.4961890243902439, 0.49254843517138597,
    0.49080206033848417, 0.5052473763118441, 0.4891385767790262,
    0.49623493975903615, 0.5023041474654378, 0.49736247174076864,
    0.5135542168674698, 0.5041075429424944, 0.4894578313253012,
    0.4878048780487805, 0.3777027027027027, 0.4836309523809524,
    0.5026656511805027, 0.45508100147275404, 0.49181547619047616,
    0.4880952380952381, 0.5022865853658537, 0.5041384499623778,
    0.49923195084485406, 0.49849624060150377, 0.4817073170731707,
    0.48861911987860396, 0.4802680565897245, 0.4950943396226415,
], dtype=torch.float64)


TRATAMIENTOS = {
    "TF-IDF": TFIDF,
    "BERT": BERT,
    "SmolLM3 sin shots": SMOLLM3_SIN_SHOTS,
    "SmolLM3 2 shots": SMOLLM3_2_SHOTS,
    "Phi-4 sin shots": PHI4_SIN_SHOTS,
    "Phi-4 2 shots": PHI4_2_SHOTS,
}


def ajustar_holm(p_values):
    ordenados = sorted(enumerate(p_values), key=lambda x: x[1])
    ajustados = [0.0] * len(p_values)
    maximo_acumulado = 0.0
    m = len(p_values)

    for rank, (indice_original, p_value) in enumerate(ordenados):
        p_ajustado = min((m - rank) * p_value, 1.0)
        maximo_acumulado = max(maximo_acumulado, p_ajustado)
        ajustados[indice_original] = maximo_acumulado

    return ajustados


def obtener_rangos_y_empates(matriz):
    orden = torch.argsort(matriz, dim=1)
    rangos = torch.empty_like(matriz)
    suma_empates = torch.tensor(0.0, dtype=matriz.dtype)

    for fila in range(matriz.shape[0]):
        valores_fila = matriz[fila]
        orden_fila = orden[fila]
        posicion = 0

        while posicion < len(orden_fila):
            siguiente = posicion + 1
            valor = valores_fila[orden_fila[posicion]]
            while siguiente < len(orden_fila) and valores_fila[orden_fila[siguiente]] == valor:
                siguiente += 1

            rango_promedio = (posicion + 1 + siguiente) / 2
            rangos[fila, orden_fila[posicion:siguiente]] = rango_promedio
            grupo = siguiente - posicion
            if grupo > 1:
                suma_empates += grupo ** 3 - grupo
            posicion = siguiente

    return rangos, suma_empates


def friedman_torch(matriz):
    n_corridas, k_tratamientos = matriz.shape
    # Como la metrica es accuracy, mayor es mejor. Se rankea -valor para que
    # rango 1 signifique mejor tratamiento.
    rangos, suma_empates = obtener_rangos_y_empates(-matriz)
    suma_rangos = rangos.sum(dim=0)
    estadistico_sin_correccion = (
        12
        / (n_corridas * k_tratamientos * (k_tratamientos + 1))
        * torch.sum(suma_rangos ** 2)
        - 3 * n_corridas * (k_tratamientos + 1)
    )

    correccion_empates = 1 - suma_empates / (
        n_corridas * k_tratamientos * (k_tratamientos ** 2 - 1)
    )
    estadistico = estadistico_sin_correccion / correccion_empates
    return estadistico, rangos.mean(dim=0)


def main():
    nombres = list(TRATAMIENTOS)
    valores = [TRATAMIENTOS[nombre] for nombre in nombres]
    n_corridas = 30

    matriz = torch.stack(valores, dim=1)
    estadistico, rangos = friedman_torch(matriz)
    p_value = chi2.sf(estadistico.item(), df=len(valores) - 1)
    kendall_w = estadistico / (n_corridas * (len(valores) - 1))

    print("Resumen por tratamiento")
    print("-" * 72)
    for nombre, vector, rango in zip(nombres, valores, rangos):
        print(
            f"{nombre:20s} media={vector.mean().item():.6f} "
            f"std={vector.std(unbiased=True).item():.6f} rango_medio={rango.item():.3f}"
        )

    print("\nPrueba de Friedman")
    print("-" * 72)
    print(f"chi2 = {estadistico.item():.6f}")
    print(f"p    = {p_value:.6g}")
    print(f"W    = {kendall_w.item():.6f}")

    comparaciones = []
    p_values = []
    for i, j in combinations(range(len(nombres)), 2):
        resultado = wilcoxon(
            valores[i].detach().cpu().numpy(),
            valores[j].detach().cpu().numpy(),
            zero_method="wilcox",
        )
        comparaciones.append((nombres[i], nombres[j], resultado.statistic, resultado.pvalue))
        p_values.append(resultado.pvalue)

    p_holm = ajustar_holm(p_values)

    print("\nPost-hoc Wilcoxon pareado con correccion Holm")
    print("-" * 72)
    for (a, b, w, p), p_ajustado in zip(comparaciones, p_holm):
        significativo = "si" if p_ajustado < 0.05 else "no"
        print(
            f"{a:20s} vs {b:20s} "
            f"W={w:7.3f} p={p:.6g} p_holm={p_ajustado:.6g} sig={significativo}"
        )


if __name__ == "__main__":
    main()
