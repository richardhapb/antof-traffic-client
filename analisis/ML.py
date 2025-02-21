# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import utils
import pandas as pd
import numpy as np

alerts, jams = utils.load_data()

# %%
alerts.shape

# %%
alerts.head()

# %%
alerts_cleaned = utils.extract_event(alerts, ["JAM", "ACCIDENT"], extra_col=['type']).drop('uuid', axis=1)
alerts_cleaned['day'] = alerts_cleaned.inicio.dt.day
alerts_cleaned['month'] = alerts_cleaned.inicio.dt.month
alerts_cleaned['year'] = alerts_cleaned.inicio.dt.year
alerts_cleaned['minute'] = alerts_cleaned.inicio.dt.minute
alerts_cleaned = alerts_cleaned.drop(['inicio', 'fin', 'geometry'], axis=1)

alerts_cleaned.head()

# %%
alerts_cleaned.info()

# %%
print(f"{len(alerts_cleaned[alerts_cleaned['street'].isna()]) / len(alerts_cleaned['street']) * 100:.2f}% de lo datos es nulos en 'street'")

# %%
alerts_cleaned["street"].value_counts()[:10]

# %%
alerts_cleaned["street"].value_counts().index[:10]

# %%
# Eliminamos valores nulos y filtro por calle

streets = alerts_cleaned["street"].value_counts().index[:10].to_numpy()

alerts_cleaned = alerts_cleaned[alerts_cleaned['street'].apply(lambda x: x in streets)]
alerts_cleaned.info()

# %%
cat = ["street", "type", "day_type"]
num = list(alerts_cleaned.drop(cat, axis=1).columns)

# %% [markdown]
# ## Balanceo de eventos (creación de no-eventos ficticios)

# %%
# Balanceo de eventos y creación de no eventos

events = alerts_cleaned.copy()
events['happen'] = 1

q_events = len(events)

street = events['street']
x = events['x']
y = events['y']
type = np.random.choice(["ACCIDENT", "JAM"], q_events)
hour = np.random.randint(events.hour.min(), events.hour.max(), q_events)
minute = np.random.randint(events.minute.min(), events.minute.max(), q_events)
week_day = np.random.randint(events.week_day.min(), events.week_day.max(), q_events)
day_type = np.random.choice(["s", "f"], q_events)
day = np.random.randint(events.day.min(), events.day.max() + 1, q_events)
month = np.random.randint(events.month.min(), events.month.max() + 1, q_events)
year = np.random.choice(events.year.unique(), q_events)


no_events = pd.DataFrame({
    "street": street,
    "x": x,
    "y": y,
    "type": type,
    "hour": hour,
    "minute": minute,
    "week_day": week_day,
    "day_type": day_type,
    "day": day,
    "month": month,
    "year": year,
    "happen": 0
})

no_events

# %%
total_events = pd.concat([events, no_events], axis=0)
total_events['happen'].value_counts()

# %% [markdown]
# ## Ocurrencia de evento

# %%
# Se elimina mes y año porque no hay muestras suficientes

X_happen = total_events.drop(['happen', "type", "x", "y", "street", "month", "year"], axis=1)
y_happen = total_events['happen']

# %%
dt = {"f": 0, "s": 1}

X_happen["day_type"] = X_happen["day_type"].map(dt)


# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X_train_happen, X_test_happen, y_train_happen, y_test_happen = train_test_split(X_happen, y_happen, test_size=0.2, random_state=42)

rfc_happen = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10, min_samples_split=5, n_estimators=100)
rfc_happen.fit(X_train_happen, y_train_happen)

y_predict_happen = rfc_happen.predict(X_test_happen)

# %%
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score

confusion_matrix(y_test_happen, y_predict_happen)

# %%
recall = recall_score(y_test_happen, y_predict_happen)
precision = precision_score(y_test_happen, y_predict_happen)
f1 = f1_score(y_test_happen, y_predict_happen)

recall, precision, f1

# %%
# Un dato de prueba

test = {
    "hour": [7],
    "week_day": [4],
    "day_type": [1],
    "day": [11],
    "minute": [20]
}

rfc_happen.predict_proba(pd.DataFrame(test)), rfc_happen.predict(pd.DataFrame(test))


# %%
from sklearn.model_selection import cross_val_score

cve_rfc_happen = cross_val_score(rfc_happen, X_happen, y_happen, cv=10)
cve_rfc_happen

# %%
mean_cve_rfc_happen = np.mean(cve_rfc_happen)
mean_cve_rfc_happen

# %% [markdown]
# ## Tipo de evento

# %%
X_type = events.drop(["happen", "month", "year", "x", "y", "type", "street"], axis=1)
y_type = events["type"]

# %%
X_type["day_type"] = X_type["day_type"].map(dt)

# %%
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

le_type = LabelEncoder()
y_type = le_type.fit_transform(y_type)



# %%
X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(X_type, y_type, test_size=0.2, random_state=42)

# %%
rfc_type = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20, class_weight="balanced", min_samples_split=10, max_samples=20)
rfc_type.fit(X_train_type, y_train_type)

y_predict_type = rfc_type.predict(X_test_type)

# %%
cve_rfc_type = cross_val_score(rfc_type, X_type, y_type, cv=10)
cve_rfc_type

# %%
mean_cve_rfc_type = np.mean(cve_rfc_type)
mean_cve_rfc_type

# %%
confusion_matrix(y_test_type, y_predict_type)

# %%
recall = recall_score(y_test_type, y_predict_type)
precision = precision_score(y_test_type, y_predict_type)
f1 = f1_score(y_test_type, y_predict_type)

recall, precision, f1

# %%
events['type'].value_counts()

# %% [markdown]
# ## Calle de evento

# %%
X_street = events.drop(["happen", "month", "year", "x", "y", "street"], axis=1)
y_street = events['street']

# %%
le_street = LabelEncoder()

for c in cat:
    if c in X_street.columns:
        X_street[c] = le_street.fit_transform(X_street[c])

y_street = le_street.fit_transform(y_street)

# %%
sm_street = SMOTE(random_state=42)

X_street_resampled, y_street_resampled = sm_street.fit_resample(X_street, y_street)

X_train_street, X_test_street, y_train_street, y_test_street = train_test_split(X_street_resampled, y_street_resampled, test_size=0.2, random_state=42)
rfc_street = RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced")
rfc_street.fit(X_train_street, y_train_street)

# %%
y_predict_street = rfc_street.predict(X_test_street)

# %%
cve_rfc_street = cross_val_score(rfc_street, X_street, y_street, cv=10)
cve_rfc_street

# %%
mean_cve_rfc_street = np.mean(cve_rfc_street)
mean_cve_rfc_street

# %%
confusion_matrix(y_test_street, y_predict_street)

# %%
recall = recall_score(y_test_street, y_predict_street, average='weighted')
precision = precision_score(y_test_street, y_predict_street, average='weighted')
f1 = f1_score(y_test_street, y_predict_street, average='weighted')

recall, precision, f1

# %%
# Un dato de prueba

happen = ["No ocurre", "Ocurre"]

test = {
    "hour": [7],
    "week_day": [4],
    "day_type": [1],
    "day": [10],
    "minute": [20]
}

rfc_happen.predict_proba(pd.DataFrame(test)), happen[rfc_happen.predict(pd.DataFrame(test))[0]]


# %%
rfc_type.predict_proba(pd.DataFrame(test)), le_type.classes_[rfc_type.predict(pd.DataFrame(test))][0]

# %%
test_street = pd.concat([pd.DataFrame({"type": [rfc_type.predict(pd.DataFrame(test))[0]]}), pd.DataFrame(test)], axis = 1)

# %%
rfc_street.predict_proba(pd.DataFrame(test_street)), le_street.classes_[rfc_street.predict(pd.DataFrame(test_street))[0]]

# %% [markdown]
# Ocurrirá un evento tipo JAM en Av. Edmundo Pérez Zujovic el 10 de Octubre a las 7:20.

# %%
# Distribución de probabilidad de evento durante el día

prob_pred_rfc = []

for h in range(0, 24):
    for m in range(0, 60):
        pred = rfc_happen.predict_proba(pd.DataFrame({
                "hour": [h],
                "week_day": [4],
                "day_type": [1],
                "day": [8],
                "minute": [m]
            }))
        prob_pred_rfc.append(pred[0][1])

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 8))

# Crear la lista de etiquetas de horas y minutos cada media hora
xticks_labels = [f"{h:02}:{m:02}" for h in range(24) for m in [0, 30]]

# Crear los índices para las etiquetas cada media hora (cada 30 minutos son 2 puntos por hora)
xticks_positions = np.arange(0, len(prob_pred_rfc), 30)

plt.plot(prob_pred_rfc)
plt.hlines(0.5, 0, len(prob_pred_rfc), colors="r")
plt.xticks(xticks_positions, xticks_labels, rotation=45)

# Mostrar el gráfico
plt.xlabel("Tiempo (Hora:Minuto)")
plt.ylabel("Probabilidad de evento")
plt.title("Probabilidad de que un evento ocurra durante el día")
plt.show()

# %% [markdown]
# ## MLP happen

# %%
from sklearn.neural_network import MLPClassifier

mlp_happen = MLPClassifier(activation="tanh", random_state=42, learning_rate_init=0.01, max_iter=300, hidden_layer_sizes=(40,))

mlp_happen.fit(X_train_happen, y_train_happen)


# %%
y_predict_happen_mlp = mlp_happen.predict(X_test_happen)

# %%
confusion_matrix(y_predict_happen_mlp, y_test_happen)

# %%
recall = recall_score(y_predict_happen_mlp, y_test_happen)
precision = precision_score(y_predict_happen_mlp, y_test_happen)
f1 = f1_score(y_predict_happen_mlp, y_test_happen)

recall, precision, f1

# %% [markdown]
# RandomForest
#
#
# (np.float64(0.9592592592592593),
#  np.float64(0.8345864661654135),
#  np.float64(0.8925904652498564))

# %%
cve_mlp_happen = cross_val_score(mlp_happen, X_happen, y_happen, cv=10)
cve_mlp_happen

# %%
mean_cve_mlp_happen = np.mean(cve_mlp_happen)
mean_cve_mlp_happen

# %% [markdown]
# ## MLP Type

# %%
mlp_type = MLPClassifier(activation="logistic", random_state=42, learning_rate_init=0.01, max_iter=300, hidden_layer_sizes=(40,))

mlp_type.fit(X_train_type, y_train_type)
y_predict_type_mlp = mlp_type.predict(X_test_type)

# %%
confusion_matrix(y_predict_type_mlp, y_test_type)

# %%
recall = recall_score(y_predict_type_mlp, y_test_type)
precision = precision_score(y_predict_type_mlp, y_test_type)
f1 = f1_score(y_predict_type_mlp, y_test_type)

recall, precision, f1

# %% [markdown]
# Random Forest
#
# (np.float64(1.0),
#  np.float64(0.8952496954933008),
#  np.float64(0.9447300771208226))

# %%
cve_mlp_type = cross_val_score(mlp_type, X_type, y_type, cv=10)
cve_mlp_type

# %%
mean_cve_mlp_type = np.mean(cve_mlp_type)
mean_cve_mlp_type

# %% [markdown]
# ## MLP Street

# %%
mlp_street = MLPClassifier(activation="tanh", random_state=42, learning_rate_init=0.01, max_iter=300, hidden_layer_sizes=(300,))

mlp_street.fit(X_train_street, y_train_street)
y_predict_street_mlp = mlp_street.predict(X_test_street)

# %%
confusion_matrix(y_predict_street_mlp, y_test_street)

# %%
recall = recall_score(y_predict_street_mlp, y_test_street, average="weighted")
precision = precision_score(y_predict_street_mlp, y_test_street, average="weighted")
f1 = f1_score(y_predict_street_mlp, y_test_street, average="weighted")

recall, precision, f1

# %% [markdown]
# Random Forest
#
# (np.float64(0.7139053254437869),
#  np.float64(0.7062531483583431),
#  np.float64(0.7086289747169453))

# %%
cve_mlp_street = cross_val_score(mlp_street, X_street, y_street, cv=10)
cve_mlp_street

# %%
mean_cve_mlp_street = np.mean(cve_mlp_street)
mean_cve_mlp_street

# %% [markdown]
# ## XGBClassifier happen

# %%
from xgboost import XGBClassifier

xgb_happen = XGBClassifier(learning_rate=0.03, random_state=42, n_estimators=50, max_depth=5, gamma=0.2, colsample_bytree=0.7)
xgb_happen.fit(X_train_happen, y_train_happen)

# %%
y_predict_happen_xgb = xgb_happen.predict(X_test_happen)

# %%
confusion_matrix(y_predict_happen_xgb, y_test_happen)

# %%
recall = recall_score(y_predict_happen_xgb, y_test_happen)
precision = precision_score(y_predict_happen_xgb, y_test_happen)
f1 = f1_score(y_predict_happen_xgb, y_test_happen)

recall, precision, f1

# %% [markdown]
# Random Forest
#
# (np.float64(0.9592592592592593),
#  np.float64(0.8345864661654135),
#  np.float64(0.8925904652498564))

# %%
cve_xgb_happen = cross_val_score(xgb_happen, X_happen, y_happen, cv=10)
cve_xgb_happen

# %%
mean_cve_xgb_happen = np.mean(cve_xgb_happen)
mean_cve_xgb_happen

# %% [markdown]
# ## XGBClassifier type

# %%
xgb_type = XGBClassifier(learning_rate=0.03, random_state=42, n_estimators=50, max_depth=5, gamma=0.2, colsample_bytree=0.7)
xgb_type.fit(X_train_type, y_train_type)
y_predict_type_xgb = xgb_type.predict(X_test_type)

# %%
confusion_matrix(y_predict_type_xgb, y_test_type)

# %%
recall = recall_score(y_predict_type_xgb, y_test_type)
precision = precision_score(y_predict_type_xgb, y_test_type)
f1 = f1_score(y_predict_type_xgb, y_test_type)

recall, precision, f1

# %% [markdown]
# Random forest
#
# (np.float64(1.0),
#  np.float64(0.8952496954933008),
#  np.float64(0.9447300771208226))

# %%
cve_xgb_type = cross_val_score(xgb_type, X_type, y_type, cv=10)
cve_xgb_type

# %%
mean_cve_xgb_type = np.mean(cve_xgb_type)
mean_cve_xgb_type

# %%
np.mean(mean_cve_xgb_type), np.mean(np.array([0.88807786, 0.88807786, 0.88780488, 0.88780488, 0.88780488,
       0.88780488, 0.88780488, 0.88780488, 0.88780488, 0.88780488]))

# %% [markdown]
# ## XGBClassifier street

# %%
xgb_street = XGBClassifier(learning_rate=0.01, random_state=42, n_estimators=20, max_depth=5, gamma=0.5, colsample_bytree=0.3)
xgb_street.fit(X_train_street, y_train_street)
y_predict_street_xgb = xgb_street.predict(X_test_street)

# %%
confusion_matrix(y_predict_street_xgb, y_test_street)

# %%
recall = recall_score(y_predict_street_xgb, y_test_street, average="weighted")
precision = precision_score(y_predict_street_xgb, y_test_street, average="weighted")
f1 = f1_score(y_predict_street_xgb, y_test_street, average="weighted")

recall, precision, f1

# %% [markdown]
# Random Forest
#
# (np.float64(0.7139053254437869),
#  np.float64(0.7062531483583431),
#  np.float64(0.7086289747169453))

# %%
cve_xgb_street = cross_val_score(xgb_street, X_street, y_street, cv=10)
cve_xgb_street

# %%
mean_cve_xgb_street = np.mean(cve_xgb_street)
mean_cve_xgb_street

# %%
mean_cve_rfc_happen, mean_cve_mlp_happen, mean_cve_xgb_happen

# %%
mean_cve_rfc_type, mean_cve_mlp_type, mean_cve_xgb_type

# %%
mean_cve_rfc_street, mean_cve_mlp_street, mean_cve_xgb_street

# %%
# Distribución de probabilidad de evento durante el día

prob_pred_xgb = []

for h in range(0, 24):
    for m in range(0, 60):
        pred = xgb_happen.predict_proba(pd.DataFrame({
                "hour": [h],
                "week_day": [4],
                "day_type": [1],
                "day": [8],
                "minute": [m]
            }))
        prob_pred_xgb.append(pred[0][1])

# %%
plt.figure(figsize=(16, 8))

# Crear la lista de etiquetas de horas y minutos cada media hora
xticks_labels = [f"{h:02}:{m:02}" for h in range(24) for m in [0, 30]]

# Crear los índices para las etiquetas cada media hora (cada 30 minutos son 2 puntos por hora)
xticks_positions = np.arange(0, len(prob_pred_xgb), 30)

plt.plot(prob_pred_xgb)
plt.hlines(0.5, 0, len(prob_pred_xgb), colors="r")
plt.xticks(xticks_positions, xticks_labels, rotation=45)

# Mostrar el gráfico
plt.xlabel("Tiempo (Hora:Minuto)")
plt.ylabel("Probabilidad de evento")
plt.title("Probabilidad de que un evento ocurra durante el día")
plt.show()
