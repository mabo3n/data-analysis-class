# --- Dependencias e configuracoes gerais ---


# Importa bibliotecas necessarias
import numpy as np                       # Computacao numerica
import matplotlib.pyplot as plt          # Plotagem de graficos
import pandas as pd                      # Manipulacao de datasets
from scipy.stats import linregress       # Regressao linear
from scipy.optimize import curve_fit     # Regressao nao-linear

# Configura para nao cortar exibicao de datasets no terminal
pd.options.display.max_columns = None
pd.options.display.width = 200


# --- Leitura e analise inicial do dataset ---


# Le arquivo em um dataframe
path = './covid19-Mar-18-2020.csv'
data = pd.read_csv(path, sep=',', decimal='.', quotechar='\'')

# Visao geral dos dados
data.head(10)
data.describe(include='all')

# Converte coluna de data para datetime
data.date = data.date.apply(pd.to_datetime)

# Exibe correlacao de casos confirmados com populacao estimada
data[['confirmed', 'estimated_population_2019']].corr()

# Exibe grafico de dispersao entre casos confirmados e populacao estimada
plt.scatter(data.estimated_population_2019, data.confirmed)
plt.show()


# --- Analise dos dados de floripa ---


# Filtro de dados onde a cidade contem "Floria", ignorando NA
floripa = data[data.city.str.contains('Floria', na=False)]

# Visao geral dos dados de Floripa
floripa.head()
floripa.describe()

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


# --- Define funcoes dos modelos ---


def linear_fit(slope, intercept):
    return lambda x: intercept + slope * x


# --- Predicoes com os modelos ---


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
