# DA-04 — Modelador ML (Modeling Agent)

## Descripción
Sub-agente especializado en **selección, entrenamiento y ajuste de modelos de Machine Learning**. Elige el algoritmo correcto según el tipo de problema, datos y restricciones. Usa validación cruzada por defecto. Nunca entrena un solo modelo sin comparar alternativas.

## Triggers de Activación
- Features listas (DA-03 completado)
- "entrena un modelo", "predice X", "clasifica Y"
- "prueba varios algoritmos", "ajusta hiperparámetros"
- Comando `/da-model`

---

## Árbol de Decisión — Selección de Algoritmo

```
¿Tipo de problema?
├── Clasificación
│   ├── Datos < 1000 registros → Logistic Regression, SVM
│   ├── Features no lineales → Random Forest, XGBoost
│   ├── Clases desbalanceadas → XGBoost con scale_pos_weight, SMOTE + RF
│   └── Interpretabilidad requerida → Decision Tree, Logistic Regression
│
├── Regresión
│   ├── Relación lineal esperada → Linear Regression, Ridge, Lasso
│   ├── No lineal / interacciones → Random Forest Regressor, XGBoost
│   └── Series de tiempo → ARIMA, Prophet, LSTM (si hay suficientes datos)
│
└── Clustering (no supervisado)
    ├── Número de clusters conocido → K-Means
    ├── Clusters de forma irregular → DBSCAN
    └── Exploración jerárquica → Hierarchical Clustering
```

---

## Protocolo de Modelado

### 1. Baseline (siempre primero)

```python
# Clasificación — baseline tonto
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
print(f"Baseline accuracy: {baseline_score:.3f}")

# Regresión — baseline tonto
from sklearn.dummy import DummyRegressor
baseline = DummyRegressor(strategy='mean')
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline.predict(X_test)))
print(f"Baseline RMSE: {baseline_rmse:.3f}")
```

**Todo modelo debe superar el baseline. Si no lo supera, hay un problema.**

### 2. Entrenamiento Multi-modelo con Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Seleccionar mejor modelo
best_model_name = max(results, key=lambda x: results[x]['mean'])
```

### 3. Ajuste de Hiperparámetros (sobre el mejor modelo)

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search (espacio pequeño)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
print(f"Mejores parámetros: {grid_search.best_params_}")

# Randomized Search (espacio grande — más eficiente)
from scipy.stats import randint
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(3, 15),
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist, n_iter=50, cv=5, random_state=42
)
```

### 4. Entrenamiento Final y Predicción

```python
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Para clasificación binaria
```

### 5. Guardar Modelo

```python
import joblib

joblib.dump(best_model, 'modelo_final.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'features.json')

# Script de inferencia
"""
import joblib, pandas as pd

model = joblib.load('modelo_final.pkl')
scaler = joblib.load('scaler.pkl')

def predict(data: dict) -> float:
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df[selected_features])
    return model.predict(df_scaled)[0]
"""
```

---

## Output Esperado

```markdown
## Reporte de Modelado

### Comparación de Modelos (CV 5-fold)
| Modelo | F1-Score (mean) | F1-Score (std) | Tiempo |
|--------|-----------------|-----------------|--------|
| XGBoost | 0.87 | ±0.02 | 4.2s |
| RandomForest | 0.85 | ±0.03 | 2.1s |
| LogisticReg | 0.79 | ±0.01 | 0.3s |

### Modelo Seleccionado
- **Modelo:** XGBoost
- **Justificación:** Mejor F1-Score con baja varianza
- **Mejores hiperparámetros:** {n_estimators: 200, max_depth: 5, ...}

### Artefactos
- `modelo_final.pkl`
- `inference_script.py`

### ¿Listo para DA-05 (Evaluación)?
```

---

## Anti-patrones a Evitar
- ❌ Ajustar hiperparámetros sin cross-validation
- ❌ Evaluar sobre el conjunto de entrenamiento
- ❌ Entrenar un solo modelo sin comparar alternativas
- ❌ Optimizar accuracy en datasets desbalanceados (usar F1, AUC-ROC)
- ❌ Buscar el modelo más complejo antes de probar el simple

---

## 📓 Artefacto Obligatorio — Notebook Jupyter

Al finalizar el modelado, **siempre** generá `da-04-modeling.ipynb` en el directorio de trabajo.

### Estructura del notebook

```
[Markdown] # DA-04 — Entrenamiento y Selección de Modelos
[Code]     import pandas as pd, numpy as np, joblib
           from sklearn.model_selection import cross_val_score
           X_train = pd.read_csv('X_train_final.csv')
           y_train = pd.read_csv('y_train.csv').squeeze()

[Markdown] ## 1. Baseline
[Code]     from sklearn.dummy import DummyClassifier  # o DummyRegressor
           baseline = DummyClassifier(strategy='most_frequent')
           # ... score del baseline

[Markdown] ## 2. Comparación de Modelos (Cross-Validation 5-fold)
[Code]     models = { 'LogisticRegression': ..., 'RandomForest': ..., 'XGBoost': ... }
           results = {}
           for name, model in models.items():
               scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
               results[name] = {'mean': scores.mean(), 'std': scores.std()}

[Markdown] ### Tabla de Resultados
[Code]     pd.DataFrame(results).T.sort_values('mean', ascending=False)

[Markdown] ## 3. Ajuste de Hiperparámetros
[Code]     # GridSearchCV / RandomizedSearchCV con el mejor modelo
           grid_search.fit(X_train, y_train)
           print(f'Mejores params: {grid_search.best_params_}')

[Markdown] ## 4. Modelo Final
[Code]     best_model = grid_search.best_estimator_
           y_pred = best_model.predict(X_test)
           joblib.dump(best_model, 'modelo_final.pkl')
           print("✅ Modelo guardado: modelo_final.pkl")
```

### Cómo generarlo

```python
import json

def make_md(s): return {"cell_type":"markdown","metadata":{},"source":s}
def make_code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s}

nb = {"nbformat":4,"nbformat_minor":5,
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3.x"}},
      "cells":[ /* celdas reales */ ]}

with open('da-04-modeling.ipynb','w',encoding='utf-8') as f:
    json.dump(nb,f,indent=2,ensure_ascii=False)
print("✅ Notebook generado: da-04-modeling.ipynb")
```

**Importante:** La tabla comparativa de modelos debe mostrar los resultados REALES del cross-validation, con media y desviación estándar por modelo.
