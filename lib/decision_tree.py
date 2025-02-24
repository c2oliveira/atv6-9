data_veiculos = {
    'potencia': [100, 150, 130, 90, 120, 170, 110, 160, 140, 95, 105, 155, 135, 115, 125, 145, 165, 175, 185, 200],
    'peso':     [1200, 1500, 1300, 1100, 1250, 1600, 1150, 1550, 1400, 1050, 1180, 1520, 1380, 1120, 1260, 1420, 1580, 1650, 1700, 1900]
}
df_veiculos = pd.DataFrame(data_veiculos)

# Rótulos gerados com base em uma regra simples:
# Veículos com potência < 130 e peso < 1300 são considerados "Econômicos"
df_veiculos['classe'] = df_veiculos.apply(
    lambda row: 'Econômico' if (row['potencia'] < 130 and row['peso'] < 1300) else 'Não Econômico', axis=1)

# Converter os rótulos para numérico para treinamento (1 = Econômico, 0 = Não Econômico)
df_veiculos['classe_num'] = df_veiculos['classe'].map({'Econômico': 1, 'Não Econômico': 0})

X_veiculos = df_veiculos[['potencia', 'peso']]
y_veiculos = df_veiculos['classe_num']

# Treinamento da Árvore de Decisão
modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_veiculos, y_veiculos)

print("\nÁrvore de Decisão - Classificação de Veículos")
print("Acurácia no conjunto de treinamento:", modelo_arvore.score(X_veiculos, y_veiculos))

# Visualização da árvore de decisão
plt.figure(figsize=(12,8))
plot_tree(modelo_arvore, feature_names=['potencia', 'peso'],
          class_names=['Não Econômico', 'Econômico'], filled=True)
plt.title("Árvore de Decisão - Classificação de Veículos")
plt.show()
