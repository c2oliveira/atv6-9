import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importações para modelos de ML
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set_style("whitegrid")
data_heart = {
    'horas_atividade': [0, 1, 2, 3, 4, 5, 1.5, 2.5, 3.5, 4.5, 5.5, 6, 0.5, 7, 8, 2, 3, 4, 5, 6],
    'risco':           [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
}
df_heart = pd.DataFrame(data_heart)
X_heart = df_heart[['horas_atividade']]
y_heart = df_heart['risco']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_heart, y_heart, test_size=0.3, random_state=42)

# Treinamento do modelo de regressão logística
modelo_log = LogisticRegression()
modelo_log.fit(X_train, y_train)
y_pred_log = modelo_log.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

print("\nRegressão Logística - Previsão do Risco de Doença Cardíaca")
print("Acurácia:", acc_log)

# Visualizar a curva sigmoide sobre o intervalo de horas
x_range = np.linspace(0, 8, 300).reshape(-1, 1)
y_prob = modelo_log.predict_proba(x_range)[:,1]
plt.figure(figsize=(8,5))
plt.scatter(X_heart, y_heart, color='black', label='Dados')
plt.plot(x_range, y_prob, color='green', label='Curva Sigmoide')
plt.title("Atividade Física vs. Risco de Doença Cardíaca")
plt.xlabel("Horas de Atividade Física por Semana")
plt.ylabel("Probabilidade de Risco")
plt.legend()
plt.show()
