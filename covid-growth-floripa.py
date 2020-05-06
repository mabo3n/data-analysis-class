import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit

pd.options.display.max_columns = None
pd.options.display.width = 200

path = './covid19-Mar-18-2020.csv'
data = pd.read_csv(path, sep=',', decimal='.', quotechar='\'')

data.head(10)
data.describe(include='all')

data.date = pd.to_datetime(data.date)

data[['confirmed', 'estimated_population_2019']].corr()
plt.scatter(data.estimated_population_2019, data.confirmed)
plt.show()

# --- Dados de floripa ---

floripa = data[data.city.str.contains('Floria', na=False)]
floripa.head()
floripa.describe()

# Converte campo de data para datetime
floripa.date = floripa.date.apply(pd.to_datetime)

# Seta a data como o indice
# floripa.set_index('date', inplace=True)

# Antes de exibir o primeiro plot, define uma funcao utilitaria
# para formatar bonitinho o(s) ultimo(s) plots de data (eixo x)
# por casos confirmados ou preditos (eixo y)
def format_confirmed_by_date_plot(**kwargs):
    plt.xticks(rotation=20)
    plt.xlabel('Data')
    plt.ylabel('Casos confirmados')
    if kwargs.get('legend'):
        plt.legend()

# Plota casos confirmados no decorrer do tempo
plt.plot(floripa.date, floripa.confirmed, marker='o')
format_confirmed_by_date_plot()
plt.show()

# --- Define algumas funcoes dos modelos

def linear_fit(slope, intercept):
    return lambda x: intercept + slope * x

# Cria coluna id_date com representacao numerica da data
# (para poder aplicar os modelos)
floripa['id_date'] = pd.factorize(floripa.date)[0]

# Cria um modelo linear
x = floripa.id_date
y = floripa.confirmed
linear_model = linregress(x, y)

# Define uma funcao que aplica o modelo linear em um array
f = lambda x: linear_fit(linear_model.slope, linear_model.intercept)(x)

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'b.-', label='Reais')
# junto com os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), 'r.-', label='Modelo linear')
format_confirmed_by_date_plot(legend=True)
plt.show()
