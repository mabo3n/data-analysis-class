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


def linear_model(slope, intercept):
    return lambda x: intercept + slope * x

def exponential_model(a, b):
    return lambda x: a * np.exp(b * x)


# --- Predicoes com os modelos ---


# Cria coluna id_date com representacao numerica da data
# (para poder aplicar os modelos)
floripa['id_date'] = pd.factorize(floripa.date)[0]


# --- Modelo linear ---


# Cria um modelo linear
x = floripa.id_date
y = floripa.confirmed
linear_regression = linregress(x, y)

# Define uma funcao que aplica o modelo linear em um array
linear_fit = lambda x: linear_model(linear_regression.slope,
                                    linear_regression.intercept)(x)
# Cria uma alias para a funcao
f = linear_fit

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'b.-', label='Reais')
# junto com os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), 'r.-', label='Modelo linear')
format_confirmed_by_date_plot(legend=True)
plt.title('r = {}'.format(linear_regression.rvalue))
plt.show()


# --- Modelo linear do log ---


# Cria um modelo linear do log de confirmados
x = floripa.id_date
y = floripa.confirmed.apply(np.log)
linear_log_regression = linregress(x, y)

# Define uma funcao que aplica o modelo linear em um array
linear_log_fit = lambda x: linear_model(linear_log_regression.slope,
                                        linear_log_regression.intercept)(x)
# Cria um alias que aplica a funcao e exponencia o elemento (inventei a palavra)
f = lambda x: np.exp(linear_log_fit(x))

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'b.-', label='Reais')
# junto com os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), 'r.-', label='Modelo linear do log')
format_confirmed_by_date_plot(legend=True)
plt.title('r = {}'.format(linear_log_regression.rvalue))
plt.show()


# --- Modelo exponencial ---


# Define coeficientes do ultimo modelo como parametros iniciais
initial_guess = (linear_log_regression.intercept,
                  linear_log_regression.slope)

# Define uma funcao que recebe os dados e os coeficientes,
# a qual sera utilizada para ajustar o modelo
function_to_fit = lambda x, a, b: exponential_model(a, b)(x)

# Ajusta a funcao definida acima para dar fit nos dados,
# com metodo de minimos quadrados nao-linear
x = floripa.id_date
y = floripa.confirmed
estimated_coefficients, _ = curve_fit(function_to_fit, x, y, initial_guess)

# Define uma funcao que aplica o modelo exponencial
# com os coeficientes estimados em um array
exponential_fit = lambda x: (exponential_model(estimated_coefficients[0],
                                               estimated_coefficients[1]))(x)
# Cria uma alias para a funcao
f = lambda x: exponential_fit(x)

# Calcula manualmente o pearson's r (coeficiente de correlacao)
r = np.corrcoef(y, f(x))[0][1]

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'b.-', label='Reais')
# junto com os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), 'r.-', label='Modelo exponencial')
format_confirmed_by_date_plot(legend=True)
plt.title('r = {}'.format(r))
plt.show()
