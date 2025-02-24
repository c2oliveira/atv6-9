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
data_energy = {
    'num_aparelhos': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25],
    'custo':          [30, 40, 50, 65, 70, 85, 90, 100, 115, 130, 150, 170, 190, 210, 240]
}
df_energy = pd.DataFrame(data_energy)
X_energy = df_energy[['num_aparelhos']]
y_energy = df_energy['custo']

# Treinamento do modelo de regressão linear
modelo_lr = LinearRegression()
modelo_lr.fit(X_energy, y_energy)

# Previsões e avaliação com Erro Quadrático Médio (MSE)
y_pred_energy = modelo_lr.predict(X_energy)
mse_energy = mean_squared_error(y_energy, y_pred_energy)

print("Regressão Linear - Previsão do Custo de Energia")
print("Coeficiente (β1):", modelo_lr.coef_[0])
print("Intercepto (β0):", modelo_lr.intercept_)
print("Erro Quadrático Médio (MSE):", mse_energy)

# Visualização: dados reais e reta de regressão
plt.figure(figsize=(8, 5))
plt.scatter(X_energy, y_energy, color='blue', label='Dados Reais')
plt.plot(X_energy, y_pred_energy, color='red', label='Reta de Regressão')
plt.title("Custo de Energia x Número de Aparelhos")
plt.xlabel("Número de Aparelhos")
plt.ylabel("Custo Mensal (R$)")
plt.legend()
plt.show()
