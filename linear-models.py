#!/bin/python

# Importa bibliotecas necessarias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# Baixa e guarda os dados disponibilizados pelo professor em um dataframe
url = 'http://dropbox.com/s/jv9ukzk58vck3oh/Gestacao.csv?dl=1'
data = pd.read_csv(url, sep=';', decimal=',')

# Exibe os primeiros 10 registros
data.head(10)

# Exibe um resumo de cada coluna
data.describe()

x = data["Age"]
y = data["Weight"]
s = data["Births"]  # sizes

plt.xlabel("Idade gestacional (em semanas)")
plt.ylabel("Media de peso dos bebes")
plt.scatter(x, y, s)
plt.show()
