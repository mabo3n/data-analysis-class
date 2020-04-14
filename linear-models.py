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
# Define as cores (claro/escuro dependendo do número de nascimentos)
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

