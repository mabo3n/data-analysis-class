#######################################################################
##############                                             ############
##############      Script Modelos Lineares Parte II       ############
##############             15 de abril de 2020             ############
##############              Rodrigo Sant'Ana               ############
##############                                             ############
#######################################################################

###### Instalando pacotes necessarios para as analises...
install.packages(c("dplyr", "tidyr", "ggplot2", "GLMsData", "patchwork",
                   "MuMIn", "lubridate", "nlstools"),
                 dependencies = TRUE)

###### Carregando os pacotes para uso...
library(dplyr)
library(tidyr)
library(ggplot2)
library(GLMsData)
library(patchwork)
library(MuMIn)
library(lubridate)
library(nlstools)

###### Padronizando o numero de casas apos a virgula...
options(scipen = 16)

###### Carregando os dados - https://brasil.io/dataset/covid19/caso...
db <- read.table("covid19_17_04.csv", header = TRUE, sep = ",",
                 dec = ".")

###### Corrigindo os dados de data...
db$date <- ymd(db$date)

###### Vamos avaliar algumas correlacoes?

##### Qual a correlacao entre casos confirmados e populacao...
cor(db$confirmed, db$estimated_population_2019)

#### opa... temos dados perdidos na base, vamos resolver isto
#### diretamente na funcao...
cor(db$confirmed, db$estimated_population_2019,
    use = "complete.obs")

#### observando o grafico desta relacao...
p00 <- ggplot(data = db, aes(x = estimated_population_2019,
                             y = confirmed)) +
  geom_point(pch = 21, colour = "black", fill = "gray",
             alpha = 0.8, size = 5) +
  labs(x = "População", y = "Casos Confirmados")
p00

p00 <- ggplot(data = db, aes(x = estimated_population_2019,
                             y = confirmed)) +
  geom_point(pch = 21, colour = "black", fill = "gray",
             alpha = 0.8, size = 5) +
  stat_smooth(method = "lm", formula = y ~ x, se = TRUE) +
  labs(x = "População", y = "Casos Confirmados")
p00

#### E claro, temos casos variando no tempo, mas a relacao de
#### aumento e positiva, como vimos no coeficiente de correlacao.
#### Porem, vamos olhar somente para o numero total de infectados
#### no ultimo dia de medicao em cada estado...
tab01 <- db %>%
  filter(place_type == "state") %>%
  group_by(state) %>%
  summarise(populacao = mean(estimated_population_2019, na.rm = TRUE),
            infectados = max(confirmed, na.rm = TRUE)) %>%
  as.data.frame()
tab01

### Correlacao deste caso...
cor(tab01$populacao, tab01$infectados)

### Figura...
p01 <- ggplot(data = tab01, aes(x = populacao,
                                y = infectados)) +
  geom_point(pch = 21, colour = "black", fill = "gray",
             alpha = 0.8, size = 5) +
  stat_smooth(method = "lm", formula = y ~ x, se = TRUE) +
  labs(x = "População", y = "Casos Confirmados")
p01

##### E os obitos em relacao aos casos confirmados, sera que vemos
##### um padrao linear crescente...

#### Correlacao deste caso...
cor(db$confirmed, db$deaths, use = "complete.obs")

### Figura...
p02 <- ggplot(data = db, aes(x = confirmed,
                             y = deaths)) +
  geom_point(pch = 21, colour = "black", fill = "gray",
             alpha = 0.8, size = 5) +
  stat_smooth(method = "lm", formula = y ~ x, se = TRUE) +
  labs(x = "Casos Confirmados", y = "Obitos")
p02

###### Entrando em modelos lineares e não lineares...

###### Filtrando os dados dos Estados apenas...
uf <- filter(db, place_type == "state")

###### Criando uma variavel identificadora de datas...
uf$id.date <- as.numeric(as.factor(uf$date))

###### Evolucao dos casos confirmados de infeccao por COVID-19 por dia...
p03 <- ggplot(data = uf, aes(x = date, y = confirmed)) +
  geom_line() +
  geom_point(pch = 21, colour = "black", fill = "white", size = 2) +
  facet_wrap(~ state, scales = "free") +
  labs(x = "Dias", y = "Casos confirmados") +
  scale_x_date(breaks = "10 days")
p03

p04 <- ggplot(data = uf, aes(x = date, y = log(confirmed),
                             colour = state, fill = state)) +
  geom_line() +
  geom_point(pch = 21, size = 2) +
  labs(x = "Dias", y = "Casos confirmados") +
  scale_x_date(breaks = "10 days")
p04

###### Vamos isolar um unico estado e visualizar o que esta acontecendo
###### nele...

##### Filtrando os dados do Estado de SP...
sp <- filter(db, place_type == "state" & state == "SP")

#### Criando uma variavel identificadora de datas...
sp$id.date <- as.numeric(as.factor(sp$date))

#### Evolucao dos casos confirmados de infeccao por COVID-19 por dia...
p05 <- ggplot(data = sp, aes(x = date, y = confirmed)) +
  geom_line() +
  geom_point(pch = 21, colour = "black", fill = "white", size = 5) +
  labs(x = "Dias", y = "Casos confirmados") +
  scale_x_date(breaks = "10 days") +
  scale_y_continuous(limits = c(0, 15000))
p05

#### Vejam ha um crescimento exponencial dos casos confirmados, porem
#### se logaritmizarmos a escala de casos confirmados, podes ter uma
#### visao, aproximadamente, linear...
p06 <- ggplot(data = sp, aes(x = date, y = log(confirmed))) +
  geom_point(pch = 21, colour = "black", fill = "white", size = 5) +
  labs(x = "Dias", y = "Casos confirmados") +
  stat_smooth(method = "lm", formula = y ~ x) +
  scale_x_date(breaks = "10 days") +
  scale_y_continuous(limits = c(0, 15))
p06

#### Vamos modelar os casos utilizando um modelo linear para o
#### Logaritmo dos casos confirmados em SP...
mod0 <- lm(log(confirmed) ~ id.date, data = sp)

### Visualizando os coeficientes do modelo linear...
mod0

### Olhando um resumo completo dos resultados do modelo...
summary(mod0)

### predizendo os resultados com base no modelo...
sp$Pred <- exp(predict(mod0, newdata = sp, type = "response"))

### Observando os valores preditos pelo modelo em comparação com os
### casos reais...
p07 <- ggplot() +
  geom_point(data = sp, aes(x = date, y = log(confirmed)),
             pch = 21, colour = "black", fill = "white", size = 5) +
  geom_line(data = sp, aes(x = date, y = log(Pred))) +
  labs(x = "Dias", y = "Casos confirmados") +
  scale_x_date(breaks = "10 days") +
  scale_y_continuous(limits = c(0, 10))
p07

### Correlacao entre obs e pred
cor(log(sp$confirmed), log(sp$Pred))

### Com base no nosso modelo, vamos predizer o que podera acontecer
### com o numero de contaminados para os proximos 5 dias...

## Criando uma matriz de predicao...
mat.pred <- data.frame(id.date = 1:49)

## Predizendo...
mat.pred$Pred <- exp(predict(mod0, newdata = mat.pred, type = "response"))

## visualizando os casos futuros...
p08 <- ggplot() +
  geom_line(data = mat.pred, aes(x = id.date, y = Pred)) +
  geom_point(data = mat.pred, aes(x = id.date, y = Pred),
             pch = 21, colour = "black", fill = "white", size = 5) +
  geom_point(data = sp, aes(x = id.date, y = confirmed),
             pch = 21, colour = "black", fill = "red", size = 4,
             alpha = 0.5) +
  labs(x = "Dias", y = "Predição de Contágio")
p08

#### Ao invés de linearizar o ajuste, vamos utilizar modelos exponenciais
#### (Modelos não lineares) para isto...

### Parametros iniciais...
par.a <- -0.3888 # (intercepto do modelo linear)
par.b <- 0.2518 # (coeficiente angular do modelo linear)

## Ajustando um modelo exponencial...
mod1 <- nls(confirmed ~ a*exp(b*id.date), data = sp,
            start = list(a = par.a, b = par.b),
            control = nls.control(maxiter = 200))

## visualizando a saida do modelo
mod1

## predizento para nossa matriz de predicao - 5 dias a frente...
mat.pred$Pred2 <- 86.6726*exp(0.1141*mat.pred$id.date)

## estimando o intervalo de confianca...
confint2(mod1)

## predizendo o intervalo...
mat.pred$Lwr <- 53.924905*exp(0.1047572*mat.pred$id.date)
mat.pred$Upr <- 119.4203681*exp(0.1234904*mat.pred$id.date)

## visualizando os casos futuros...
p09 <- ggplot() +
  ## geom_line(data = mat.pred, aes(x = id.date, y = Pred)) +
  ## geom_point(data = mat.pred, aes(x = id.date, y = Pred),
  ##           pch = 21, colour = "black", fill = "white", size = 5) +
    geom_line(data = mat.pred, aes(x = id.date, y = Lwr),
              colour = "blue", linetype = "dashed") +
    geom_line(data = mat.pred, aes(x = id.date, y = Upr),
              colour = "blue", linetype = "dashed") +
    geom_line(data = mat.pred, aes(x = id.date, y = Pred2),
              colour = "blue") +
    geom_point(data = mat.pred, aes(x = id.date, y = Pred2),
               pch = 21, colour = "black", fill = "blue", size = 5,
               alpha = 0.5) +
    geom_point(data = sp, aes(x = id.date, y = confirmed),
               pch = 21, colour = "black", fill = "red", size = 4,
               alpha = 0.5) +
    labs(x = "Dias", y = "Predição de Contágio")
p09

##### E se tentassemos um modelo com assintota? Visto que a taxa parece
##### estar diminuindo ao final da serie observada...

##### Utilizando uma funcao geradora de valores iniciais...
SS <- getInitial(confirmed ~ SSlogis(id.date, a, xmid, taxa),
                 data = sp)

SS

##### Parametros iniciais,,,
K_start <- SS["a"]
R_start <- 1 / SS["taxa"]
N0_start <- SS["a"] / (exp(SS["xmid"] / SS["taxa"]) + 1)

##### Formula do modelo logistico - crescimento populacional logistico...
log_formula <- formula(confirmed ~ K * N0 * exp(R * id.date) /
                           (K + N0 * (exp(R * id.date)-1)))


##### Ajuste do modelo...
mod2 <- nls(log_formula, data = sp,
            start = list(K = K_start, R = R_start, N0 = N0_start),)

##### Visualizando a saida do modelo...
summary(mod2)

##### Predito vs Observado...
cor(sp$confirmed, predict(mod2))

##### Predizendo para o cenario de SP...
mat.pred$Pred3 <- predict(mod2, newdata = mat.pred)

## visualizando os casos futuros...
p10 <- ggplot() +
  geom_line(data = mat.pred, aes(x = id.date, y = Pred)) +
  geom_point(data = mat.pred, aes(x = id.date, y = Pred),
             pch = 21, colour = "black", fill = "green", size = 5) +  
    geom_line(data = mat.pred, aes(x = id.date, y = Pred2)) +
    geom_point(data = mat.pred, aes(x = id.date, y = Pred2),
              pch = 21, colour = "black", fill = "orange", size = 5) +
    geom_line(data = mat.pred, aes(x = id.date, y = Pred3),
              colour = "blue") +
    geom_point(data = mat.pred, aes(x = id.date, y = Pred3),
               pch = 21, colour = "black", fill = "blue", size = 5,
               alpha = 0.5) +
    geom_point(data = sp, aes(x = id.date, y = confirmed),
               pch = 21, colour = "black", fill = "red", size = 4,
               alpha = 0.5) +
    labs(x = "Dias", y = "Predição de Contágio")
p10
