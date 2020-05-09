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
path = './covid19-02-05-2020.csv'
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
    plt.ylabel('Número de casos')
    if kwargs.get('legend'):
        plt.legend(loc='best')

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
floripa = data[data.city.str.contains('Florianópolis', na=False)]

# Visao geral dos dados de Floripa
floripa.head()
floripa.describe()

floripa = floripa.sort_values(by='date').reset_index(drop=True)

# Plota casos confirmados no decorrer do tempo
plt.plot(floripa.date, floripa.confirmed, marker='o')
format_confirmed_by_date_plot()
plt.show()


# --- Define funcoes dos modelos ---


def linear_model(a, b):
    return lambda x: b + a * x

def exponential_model(a, b):
    return lambda x: a * np.exp(b * x)

def logistic_model(a, b, C):
    return lambda x: C / (1 + np.exp(-(a + b * x)))

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
estimated_coefs, estimated_coefs_covariance = \
    curve_fit(function_to_fit, x, y, initial_guess)

# Define uma funcao que aplica o modelo exponencial
# com os coeficientes estimados
a, b = estimated_coefs
exponential_fit = exponential_model(a, b)

# Define outras 2 funcoes que aplicam o mesmo modelo,
# no entanto com os valores minimos e maximos para o
# coeficientes (em um intervalo de 95% de confianca)
(a_lower, a_upper), (b_lower, b_upper) = \
    get_lower_and_upper_coefs(len(x),
                              estimated_coefs,
                              estimated_coefs_covariance)
exponential_lower_fit = exponential_model(a_lower, b_lower)
exponential_upper_fit = exponential_model(a_upper, b_upper)

# Cria aliases para as funcoes
f = exponential_fit
f_lwr = exponential_lower_fit
f_upr = exponential_upper_fit

# Calcula manualmente o pearson's r (coeficiente de correlacao)
r = np.corrcoef(y, f(x))[0][1]

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'o-b', label='Reais')
# e os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), '.-r', label='Modelo exponencial')
# e os valores do intervalo de confianca do modelo
plt.plot(floripa.date, f_lwr(floripa.id_date), '-r', alpha=.3)
plt.plot(floripa.date, f_upr(floripa.id_date), '-r', alpha=.3)
format_confirmed_by_date_plot(legend=True)
plt.title('r = {}'.format(r))
plt.show()


# --- Modelo logistico ---


# Captura estimativa mais recente da populacao
most_recent_estimated_population = int(
    floripa[floripa.date == floripa.date.max()]\
        .estimated_population_2019)

# Define coeficientes do penultimo modelo como parametros iniciais
initial_guess = (linear_log_regression.intercept,
                 linear_log_regression.slope,
                 most_recent_estimated_population)

# Define uma funcao que recebe um array e coeficientes,
# a qual sera utilizada para ajustar o modelo
function_to_fit = lambda x, a, b, c: logistic_model(a, b, c)(x)

# Ajusta a funcao definida acima para dar fit nos dados,
# com metodo de minimos quadrados nao-linear
x = floripa.id_date
y = floripa.confirmed
estimated_coefs, estimated_coefs_covariance = \
    curve_fit(function_to_fit, x, y, initial_guess)

# Define uma funcao que aplica o modelo logistico
# com os coeficientes estimados
a, b, c = estimated_coefs
logistic_fit = logistic_model(a, b, c)

# Define outras 2 funcoes que aplicam o mesmo modelo,
# no entanto com os valores minimos e maximos para o
# coeficientes (em um intervalo de 95% de confianca)
(a_lower, a_upper), (b_lower, b_upper), (c_lower, c_upper) = \
    get_lower_and_upper_coefs(len(x),
                              estimated_coefs,
                              estimated_coefs_covariance)
logistic_lower_fit = logistic_model(a_lower, b_lower, c_lower)
logistic_upper_fit = logistic_model(a_upper, b_upper, c_upper)

# Cria aliases para as funcoes
f = logistic_fit
f_lwr = logistic_lower_fit
f_upr = logistic_upper_fit

# Calcula manualmente o pearson's r (coeficiente de correlacao)
r = np.corrcoef(y, f(x))[0][1]

# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed, 'ob', label='Reais')
# e os valores preditos com o modelo
plt.plot(floripa.date, f(floripa.id_date), '.-r', label='Modelo logístico')
# e os valores do intervalo de confianca do modelo
plt.plot(floripa.date, f_lwr(floripa.id_date), '-r', alpha=.3)
plt.plot(floripa.date, f_upr(floripa.id_date), '-r', alpha=.3)
format_confirmed_by_date_plot(legend=True)
plt.title('Previsão com um modelo logístico')
plt.annotate('r = {}'.format(r), xy=(.6, .12), xycoords='figure fraction')
plt.show()


# --- Predicao com os modelos ---


# Cria lista de novas datas para as quais
# sera predito os casos confirmados com cada modelo
new_dates_amount = 10
last_date = floripa.date.iloc[-1]
days = lambda x: pd.Timedelta(f'{x}D')
prediction_dates = pd.date_range(last_date + days(1),
                                 last_date + days(new_dates_amount))

# Cria nova sequencia de datas e identificador numerico de datas
# incluindo datas presentes e novas datas para predicao
new_dates = [*floripa.date, *list(prediction_dates)]
new_id_dates = pd.factorize(new_dates, sort=True)[0]

# Plota os casos reais
plt.plot(floripa.date, floripa.confirmed,
         'ob', linewidth=2, label='Reais')
# e todos os modelos com as predicoes
x = new_dates
x_id = new_id_dates
plt.plot(x, linear_fit(x_id),
         '-r', alpha=.6, label='Modelo linear')
plt.plot(x, linear_log_fit(x_id),
         '-g', alpha=.6, label='Modelo linear do log de casos reais')
plt.plot(x, exponential_fit(x_id),
         '-y', alpha=.6, label='Modelo exponencial')
plt.plot(x, logistic_fit(x_id),
         '-k', alpha=.6, label='Modelo logístico')
format_confirmed_by_date_plot(legend=True)
plt.ylim((-20,500))
plt.title('Comparativo de modelos - COVID-19 Florianópolis')
plt.show()


# --- Predicao com modelo exponencial + novos casos reais ---

# Guarda novos casos reais, verificados posteriormente
future_confirmed = (('2020-05-02', 326),
                    ('2020-05-03', 347),
                    ('2020-05-04', 353),
                    ('2020-05-05', 369),
                    ('2020-05-06', 375),
                    ('2020-05-07', 371),
                    ('2020-05-08', 373))

# Cria arrays separados de novos casos e datas para plotar
future_dates, future_confirmed = zip(*future_confirmed)
future_dates = list(pd.to_datetime(future_dates))

# Recupera aliases para o modelo exponencial
# (que foi construido sem os novos valores acima)
f = exponential_fit
f_lwr = exponential_lower_fit
f_upr = exponential_upper_fit

# Define tamanho inicial para figura
plt.figure(figsize=(10,7))
# Plota os valores reais
plt.plot(floripa.date, floripa.confirmed,
         'ob', label='Confirmados')
# e os valores reais do futuro
plt.plot(future_dates, future_confirmed,
         '+b', alpha=.2, label='Confirmados futuramente')
# e os valores preditos com o modelo
x = new_dates
x_id = new_id_dates
plt.plot(x, f(x_id), '-r', label='Previstos com modelagem exponencial')
# e os valores do intervalo de confianca do modelo
plt.plot(x, f_lwr(x_id), '-r', alpha=.2)
plt.plot(x, f_upr(x_id), '-r', alpha=.2)
# e a funcao do modelo
plt.annotate(r'$y = a\it e ^{bx}$',
             xy=(.20,.25), xycoords='figure fraction',
             fontsize=12, color='red',
             bbox=dict(boxstyle="round", fc="w"))
# e o dia da reducao por casos duplicados
plt.annotate('Correção de dados duplicados', alpha=.2,
             xy=('2020-05-02', 460), xycoords='data',
             xytext=(-100, 220), textcoords='offset points',
             arrowprops=dict(arrowstyle='-|>',
                             connectionstyle="arc3,rad=-0.25",
                             facecolor='gray', alpha=.2))
# plt.text(0.5, 0.5, 'matplotlib', transform=ax.transAxes)
format_confirmed_by_date_plot(legend=True)
plt.title('01/05/2020\nPrevisão de infectados - Florianópolis (SC)')
plt.show()
