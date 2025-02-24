np.random.seed(42)
notas_math = np.concatenate([np.random.normal(60, 5, 20),
                             np.random.normal(80, 5, 15),
                             np.random.normal(50, 5, 15)])
notas_science = np.concatenate([np.random.normal(65, 5, 20),
                                np.random.normal(75, 5, 15),
                                np.random.normal(45, 5, 15)])
df_alunos = pd.DataFrame({'Matemática': notas_math, 'Ciências': notas_science})

# Aplicar K-Means definindo 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_alunos)
df_alunos['Cluster'] = clusters

print("\nK-Means - Agrupamento de Alunos")
print(df_alunos.head())

# Visualizar os clusters em um gráfico de dispersão
plt.figure(figsize=(8,5))
plt.scatter(df_alunos['Matemática'], df_alunos['Ciências'], c=df_alunos['Cluster'], cmap='viridis')
plt.title("Clusters de Alunos (Notas de Matemática e Ciências)")
plt.xlabel("Nota em Matemática")
plt.ylabel("Nota em Ciências")
plt.colorbar(label='Cluster')
plt.show()
