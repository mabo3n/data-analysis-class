#### Este script visa replicar o script disponibilizado pelo professor
#### 'script_modelos_lineares_ver00.R' na linguagem/ecossistema Python

# Importa bibliotecas necessarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# Baixa e guarda os dados disponibilizados em um dataframe
url = 'http://dropbox.com/s/jv9ukzk58vck3oh/Gestacao.csv?dl=1'
data = pd.read_csv(url, sep=';', decimal=',')

# Exibe os primeiros 10 registros
data.head(10)

# Exibe um resumo de cada coluna
data.describe()

# Constroi um grafico de dispersao
# Define os eixos
x = data.Age
y = data.Weight
# Define as cores (claro/escuro dependendo do numero de nascimentos)
colors = ['.8' if births < 20 else 'k' for births in data.Births]

# Define a descricao de cada eixo
plt.xlabel("Idade gestacional (em semanas)")
plt.ylabel("Media de peso dos bebes")
# Constroi o grafico
plt.scatter(x, y, c=colors)

# Constroi os textos com o numero de nascimento para cada ponto
np.vectorize(plt.text)(x, y + .05, s=data.Births,
                       horizontalalignment = 'center',
                       fontsize=8)
# Exibe o grafico
plt.show()

# Explicacao do modelo linear simples
# ...

# Criando vetores de dados para facilitar o processo
# x e y ja definidos acima
weights = data.Births

# Primeira tentativa de ajuste ------------------------------
# (beta0 = intercepto, beta1 = coeficiente angular)
beta0_A, beta1_A = -.9, .1

# Ajustando o modelo...
mu_A = beta0_A + beta1_A * x # Media (esta e a linha com os valores preditos)
ss_A = np.sum(weights * (y - mu_A) ** 2) # Variancia (isto e a soma dos quadrados!)
ss_A

# Constroi o grafico
plt.scatter(x, y, c=colors)
ab_line = [beta0_A + beta1_A * x_i for x_i in x]
plt.title(f'Tentativa 01 - SQR = {np.round(ss_A, 2)}')
p01 = plt.plot(x, ab_line, color='r')

# Exibe o grafico
plt.show()

# Segunda tentativa de ajuste -------------------------------
# (beta0 = intercepto, beta1 = coeficiente angular)
beta0_B, beta1_B = -3, .15

# Ajustando o modelo...
mu_B = beta0_B + beta1_B * x # Media (esta e a linha com os valores preditos)
ss_B = np.sum(weights * (y - mu_B) ** 2) # Variancia (isto e a soma dos quadrados!)
ss_B

# Constroi o grafico
plt.scatter(x, y, c=colors)
ab_line = [beta0_B + beta1_B * x_i for x_i in x]
plt.title(f'Tentativa 02 - SQR = {np.round(ss_B, 2)}')
p02 = plt.plot(x, ab_line, color='r')

# Exibe o grafico
plt.show()

# Terceira tentativa de ajuste -------------------------------
# (beta0 = intercepto, beta1 = coeficiente angular)
beta0_C, beta1_C = -2.678, .1538

# Ajustando o modelo...
mu_C = beta0_C + beta1_C * x # Media (esta e a linha com os valores preditos)
ss_C = np.sum(weights * (y - mu_C) ** 2) # Variancia (isto e a soma dos quadrados!)
ss_C

# Constroi o grafico
plt.scatter(x, y, c=colors)
ab_line = [beta0_C + beta1_C * x_i for x_i in x]
plt.title(f'Tentativa 03 - SQR = {np.round(ss_C, 2)}')
p03 = plt.plot(x, ab_line, color='r')

# Exibe o grafico
plt.show()

# Exibe a soma dos residuos de cada conjunto de parametros
print(ss_A, ss_B, ss_C)

