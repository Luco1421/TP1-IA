# Memoria de continuidad - TP2

## Contexto general

- Proyecto en `C:\Users\Luco1421\Desktop\U\ia\TPs\TP2`.
- El usuario quiere avanzar primero con documentacion y experimentos comparativos, sin tocar notebooks ni modificar `TP2.py`.
- `TP2.py` se puede revisar/importar, pero no modificar. Contiene `Dataset`, `TrainRegression`, `DEVICE`, `train_test_split`, TF-IDF base y `BERT_train`.
- Al importar `TP2.py` se ejecutan instalaciones con `pip` por `os.system`, lo que genera mucho ruido en consola; no tocar eso por ahora.
- Dataset principal: `FEINA_1.xlsx`.
- `Dataset("FEINA_1.xlsx").read()` alterna `Segment` con label `1` y `Proposal` con label `0`.

## Comparacion TF-IDF

- El archivo final actual esta en `compareTFIDF.py`.
- No tocarlo salvo que el usuario lo pida; el usuario dijo que le gusta como quedo.
- El estilo deseado es exploratorio, simple, con prints y parecido al resto del proyecto, no codigo de produccion.
- Configuracion actual observada:
  - `REGEX_NUMBERS = r"[^\W_]+"`
  - `REGEX_LETTERS = r"[a-záéíóúñü]+"`
  - `SEEDS = [101, 202, 303]`
  - `N_PAIRS = 15`
  - `N_TRIALS = 5`
  - `TRAIN_ITERATIONS = 500`
  - `EPSILON = 1e-5`
  - `FIXED_ALPHA = 0.01`
- Usa Optuna para calibrar `alpha`.
- `alpha` es learning rate de `TrainRegression`, no regularizacion.
- Rango de alpha:
  - se calcula con una cota Lipschitz aproximada:
  - `L = lambda_max(X^T X / n) / 4`
  - `alpha_max = 1 / L`
  - `alpha_min = alpha_max / 1000`
- La comparacion por semillas no encontro ganador definitivo fuerte.
- Resultado de una corrida previa con 60 textos y 3 seeds:
  - Ganadores por seed:
    - `101 -> spacy lg modo permisivo`
    - `202 -> nltk modo regex sin numeros`
    - `303 -> stanza modo estandar sin numeros`
  - Mejor MAE promedio:
    - `nltk modo regex sin numeros`, MAE calibrado promedio aprox. `0.490720`
  - Conclusion honesta:
    - no hay ganador definitivo; `nltk modo regex sin numeros` es el mejor promedio en esa prueba reducida.
- Nota conceptual:
  - usar el mismo alpha compara bajo la misma configuracion de entrenamiento;
  - calibrar alpha por preprocesador compara el mejor rendimiento razonable de cada representacion.

## Comparacion de embeddings

- El archivo actual esta en `compareEmbeddings.py`.
- Debe parecerse en estilo a `compareTFIDF.py`.
- Modelos comparados:
  - actual de `TP2.py`: `sentence-transformers/distiluse-base-multilingual-cased-v1`
  - BERT multilingual: `bert-base-multilingual-cased`
  - BETO: `dccuchile/bert-base-spanish-wwm-cased`
  - RoBERTa espanol: `PlanTL-GOB-ES/roberta-base-bne`
- Para el modelo actual de `TP2.py`, usar `SentenceTransformer(...).encode(...)`, porque ya hace pooling internamente.
- Para BERT/BETO/RoBERTa, usar `AutoModel + AutoTokenizer` y tomar `outputs.pooler_output`.
- Se elimino `mean_pooling`; el usuario cuestiono si era necesario y se simplifico usando `pooler_output`.
- Codigo actual validado con:
  - `.\.venv\Scripts\python.exe -m py_compile compareEmbeddings.py`
- No se ejecuto completo porque puede descargar modelos de Hugging Face y tardar bastante.
- Para correrlo:
  - `.\.venv\Scripts\python.exe compareEmbeddings.py`
- Si falla por red o descarga de modelos, pedir permiso de ejecucion con escalacion.

## Documentacion y prueba final 30 corridas

- Documento principal: `documentation/docu.tex`.
- El documento ya tiene secciones redactadas para:
  - justificacion de TF-IDF y preprocesamiento;
  - justificacion de embeddings como representacion alternativa;
  - LLMs zero-shot/few-shot;
  - prueba completa;
  - prueba estadistica.
- No se pudo compilar LaTeX en este entorno porque no hay `pdflatex`/`latexmk` disponible.
- Las tablas quedaron preparadas para completar manana con:
  - MAE promedio;
  - exactitud promedio;
  - desviacion estandar del MAE;
  - desviacion estandar de exactitud;
  - vocabulario promedio en TF-IDF;
  - modelo/dimension para embeddings;
  - resultados por corrida para prueba estadistica.
- Se agrego una estructura recomendada para registrar resultados por corrida:
  - seed;
  - tratamiento;
  - MAE;
  - exactitud;
  - observaciones.
- Hay pendientes intencionales con `\pendiente{...}` para llenar cuando se corran los experimentos reales.

## Script final para manana

- Archivo nuevo: `compareFinal30.py`.
- No modifica `TP2.py`, `LLMs.py` ni notebooks.
- Reutiliza de `TP2.py`:
  - `DEVICE`;
  - `Dataset`;
  - `TrainRegression`;
  - `train_test_split`.
- Ejecuta 30 corridas con particion estratificada 80/20:
  - `TF-IDF + regresion logistica`;
  - `embeddings + regresion logistica`.
- Usa `SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")` para embeddings.
- Imprime por seed:
  - indices de test;
  - resultado TF-IDF;
  - resultado embeddings.
- Guarda salidas utiles para la prueba estadistica:
  - `output/final_30_lr_results.csv`;
  - `output/final_30_partitions.csv`.
- El CSV `final_30_lr_results.csv` permite comparar tratamientos por seed usando `mae_calibrado` y `accuracy_calibrado`.
- El CSV `final_30_partitions.csv` sirve para que las corridas con LLMs usen los mismos indices de test si se quiere una comparacion pareada mas justa.
- Validado sintacticamente con:
  - `.\.venv\Scripts\python.exe -m py_compile compareFinal30.py`
- No se ejecuto la corrida completa porque el usuario la correra manana.

## Decisiones tecnicas sobre embeddings

- `AutoModel` es correcto porque se quiere extraer representaciones base y luego entrenar la regresion propia de `TP2.py`.
- No usar `AutoModelForSequenceClassification`, porque eso agregaria una cabeza de clasificacion ajena y cambiaria la comparacion.
- `pooler_output` es mas corto y valido para estos modelos, aunque no es exactamente un embedding tipo Sentence-BERT.
- Para documentacion:
  - mencionar que `distiluse-base-multilingual-cased-v1` produce embeddings de oracion con pooling interno;
  - BERT/BETO/RoBERTa se comparan usando la representacion pooled del modelo base;
  - la regresion logistica posterior se mantiene igual para aislar la calidad de la representacion.

## Preferencias del usuario para este trabajo

- No tocar notebooks.
- No tocar `TP2.py`.
- Si hay que modificar algo de notebooks o `TP2.py`, avisar para que el usuario lo haga manualmente.
- Mantener codigo exploratorio, no excesivamente robusto ni de produccion.
- Preferir prints, ejemplos y tablas por consola.
- No agregar demasiadas abstracciones si no son necesarias.
- Si se documentan resultados, ser honesto: distinguir conclusion preliminar de conclusion definitiva.
