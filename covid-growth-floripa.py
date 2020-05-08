# --- Dependencias e configuracoes gerais ---


# Importa bibliotecas necessarias
import numpy as np                       # Computacao numerica
import matplotlib.pyplot as plt          # Plotagem de graficos
import pandas as pd                      # Manipulacao de datasets
from scipy.stats import linregress       # Regressao linear
from scipy.optimize import curve_fit     # Regressao nao-linear
from scipy.stats.distributions import t  # Distribuicao de t student

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


# --- Funcoes utilitarias ---


# Funcao para formatar bonitinho o ultimo grafico criado,
# sendo ele de data no eixo x e casos confirmados ou
# preditos no eixo y
def format_confirmed_by_date_plot(**kwargs):
    plt.xticks(rotation=20)
    plt.xlabel('Data')
    plt.ylabel('Casos confirmados')
    if kwargs.get('legend'):
        plt.legend()

# Funcao que calcula e retorna valores minimos e maximos
# para os coeficientes estimados em um modelo, baseando-se
# em um intervalo de confianca de 95%.
# Este calculo e baseado no disposto neste link:
# http://kitchingroup.cheme.cmu.edu/blog/2013/02/12/Nonlinear-curve-fitting-with-parameter-confidence-intervals/
def get_lower_and_upper_coefs(data_length, coefs, coefs_covariance):

    # Nivel alpha para o intervalo. Um nivel alpha de 0.05
    # representa um intervalo de confianca de 95% (100*(1-alpha))
    alpha = 0.05

    # Calcula quantos graus de liberdade temos
    # de acordo com o tamanho da amostra e a quantidade de coeficientes
    degrees_of_freedom = max(0, data_length - len(coefs))

    # Valor na distribuicao de t student onde beiramos o
    # intervalo de confianca (no lado positivo, considerando os 2)
    t_critical_value = t.ppf(1.0-alpha/2., degrees_of_freedom)

    # Calcula o erro padrao para cada um dos parametros estimados
    # a partir da diagonal da matriz de covarianca dos coeficientes
    standard_errors = np.sqrt(np.diag(coefs_covariance))
    print(standard_errors)

    # Cria uma funcao que recece um coeficiente e seu erro padrao
    # e retorna uma tupla com os valores minimos e maximos para
    # o coeficiente
    get_coef_bounds = \
        lambda coef, std_error: (coef - std_error * t_critical_value,
                                 coef + std_error * t_critical_value)

    # Retorna o resultado da funcao acima aplicada para
    # cada um dos grupos de coeficiente/erro padrao 
    return list(map(get_coef_bounds, coefs, standard_errors))


# --- Analise dos dados de floripa ---


# Filtro de dados onde a cidade contem "Floria", ignorando NA
floripa = data[data.city.str.contains('Floria', na=False)]

# Visao geral dos dados de Floripa
floripa.head()
floripa.describe()

# Plota casos confirmados no decorrer do tempo
plt.plot(floripa.date, floripa.confirmed, marker='o')
format_confirmed_by_date_plot()
plt.show()


# --- Define funcoes dos modelos ---


def linear_model(a, b):
    return lambda x: b + a * x

def exponential_model(a, b):
    return lambda x: a * np.exp(b * x)


# --- Predicoes com os modelos ---


# Cria coluna id_date com representacao numerica da data
# (para poder aplicar os modelos)
floripa['id_date'] = pd.factorize(floripa.date, sort=True)[0]


# --- Modelo linear ---


# Cria um modelo linear
x = floripa.id_date
y = floripa.confirmed
linear_regression = linregress(x, y)

# Define uma funcao que aplica o modelo linear
# com os coeficientes encontrados
a, b = linear_regression.slope, linear_regression.intercept;
linear_fit = linear_model(a, b)
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

# Define uma funcao que aplica o modelo linear
# com os coeficientes encontrados
a, b = linear_log_regression.slope, linear_log_regression.intercept;
linear_log_fit_logscale = linear_model(a, b)
# Faz a funcao tambem converter a escala logaritmica aplicada
# para fazer a regressao, deixando os valorres na escala inicial
linear_log_fit = lambda x: np.exp(linear_log_fit_logscale(x))
# Cria uma alias para a funcao
f = linear_log_fit

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

# Define uma funcao que recebe um array e coeficientes,
# a qual sera utilizada para ajustar o modelo
function_to_fit = lambda x, a, b: exponential_model(a, b)(x)

# Ajusta a funcao definida acima para dar fit nos dados,
# com metodo de minimos quadrados nao-linear
x = floripa.id_date
y = floripa.confirmed
estimated_coefficients, _ = curve_fit(function_to_fit,
                                      x, y,
                                      initial_guess)

# Define uma funcao que aplica o modelo exponencial
# com os coeficientes estimados
a, b = estimated_coefficients
exponential_fit = exponential_model(a, b)
# Cria uma alias para a funcao
f = exponential_fit

# Calcula manualmente o pearson's r (coeficiente de correlacao)
r = np.corrcoef(y, f(x))[0][1]

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'b.-', label='Reais')
# junto com os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), 'r.-', label='Modelo exponencial')
format_confirmed_by_date_plot(legend=True)
plt.title('r = {}'.format(r))
plt.show()


# --- Visao geral ---


# Plota todos os modelos junto com os casos reais
x = floripa.date
plt.plot(x, floripa.confirmed,
         'o-b', linewidth=2, label='Reais')
plt.plot(x, linear_fit(floripa.id_date),
         '.-r', alpha=.3, label='Modelo linear')
plt.plot(x, linear_log_fit(floripa.id_date),
         '.-g', alpha=.3, label='Modelo linear do log(x)')
plt.plot(x, exponential_fit(floripa.id_date),
         '.-y', alpha=.3, label='Modelo exponencial')
format_confirmed_by_date_plot(legend=True)
plt.ylim((-20,450))
plt.title('Covid Floripa')
plt.show()
