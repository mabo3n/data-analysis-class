#######################################################################
##############                                             ############
##############           Script Modelos Lineares           ############
##############             02 de abril de 2020             ############
##############              Rodrigo Sant'Ana               ############
##############                                             ############
#######################################################################

###### Instalando pacotes necessarios para as analises...
install.packages(c("dplyr", "tidyr", "ggplot2", "GLMsData", "patchwork"), 
                 dependencies = TRUE)

###### Carregando os pacotes para uso...
library(dplyr)
library(tidyr)
library(ggplot2)
library(GLMsData)
library(patchwork)

###### Carregando os dados de um repositório no Dropbox...
db <- read.table("https://www.dropbox.com/s/jv9ukzk58vck3oh/Gestacao.csv?dl=1",
                 header = TRUE, sep = ";", dec = ",")

###### Explorando um pouco nossos dados...

##### Resumo geral...
summary(db)

##### Visualizando a distribuicao do peso em função da idade bebes...
p00 <- ggplot() +
  geom_point(data = db, aes(x = Age, y = Weight),
             pch = ifelse(db$Births < 20, 1, 19),
             size = 4) +
  geom_text(data = db, aes(x = Age, y = Weight, label = Births),
            nudge_x = 0, nudge_y = 0.2) +
  scale_y_continuous(limits = c(0, 4), expand = c(0, 0)) +
  labs(x = "Idade gestacional (semanas)", 
       y = "Média do peso de nascimento (kg)",
       caption = "Fonte: Pacote R - GLMsData")
p00

#################################### Modelo linear Simples - Y ~ X (Ambas Numéricas)

#### Assumindo que cada observação yi em nossa base de dados representa a 
#### média do peso de nascimento de "n" bebês, temos então que o peso individual
#### de cada bebê na idade gestacionai x_i tem uma variabilidade (var) específica. 
#### Deste modo, espera-se que a média y_i tem então uma variância ponderada pelo
#### número de nascimentos (m_i) em cada classe de idade gestacional var/m_i.

## Assim, nosso modelo geral pode ser descrito como:
## var(y_i) = sigma² / mi
## mu_i = beta0 + beta1 * x_i

## Buscando um ajuste na mão - sem interação o que temos é:

## Apenas para facilitar o processo, vamos criar alguns vetores de dados...
y <- db$Weight
x <- db$Age
wts <- db$Births

## Chutes iniciais para os parametros...

# 1) Primeira tentativa de ajuste
beta0.A <- -0.9; beta1.A <- 0.1

# Ajustando o modelo...
mu.A <- beta0.A + beta1.A * x ## Média
SA <- sum(wts*(y - mu.A)^2) ## Variância
SA

# Gráfico...
p01 <- ggplot() +
  geom_point(data = db, aes(x = Age, y = Weight),
             pch = ifelse(db$Births < 20, 1, 19),
             size = 4) +
  geom_abline(slope = beta1.A, intercept = beta0.A, color = "red") +
  scale_y_continuous(limits = c(0, 4), expand = c(0, 0)) +
  labs(x = "Idade gestacional (semanas)", 
       y = "Média do peso de nascimento (kg)",
       caption = "Fonte: Pacote R - GLMsData",
       title = paste0("Tentativa 01 - SQR = ", round(SA, 2)))
p01

# 2) Segunda tentativa de ajuste
beta0.B <- -3; beta1.B <- 0.15

# Ajustando o modelo...
mu.B <- beta0.B + beta1.B * x ## Média
SB <- sum(wts*(y - mu.B)^2) ## Variância
SB

# Gráfico...
p02 <- ggplot() +
  geom_point(data = db, aes(x = Age, y = Weight),
             pch = ifelse(db$Births < 20, 1, 19),
             size = 4) +
  geom_abline(slope = beta1.B, intercept = beta0.B, color = "red") +
  scale_y_continuous(limits = c(0, 4), expand = c(0, 0)) +
  labs(x = "Idade gestacional (semanas)", 
       y = "Média do peso de nascimento (kg)",
       caption = "Fonte: Pacote R - GLMsData",
       title = paste0("Tentativa 02 - SQR = ", round(SB, 2)))
p02

# 3) Terceira tentativa de ajuste
beta0.C <- -2.678; beta1.C <- 0.1538

# Ajustando o modelo...
mu.C <- beta0.C + beta1.C * x ## Média
SC <- sum(wts*(y - mu.C)^2) ## Variância
SC

# Gráfico...
p03 <- ggplot() +
  geom_point(data = db, aes(x = Age, y = Weight),
             pch = ifelse(db$Births < 20, 1, 19),
             size = 4) +
  geom_abline(slope = beta1.C, intercept = beta0.C, color = "red") +
  scale_y_continuous(limits = c(0, 4), expand = c(0, 0)) +
  labs(x = "Idade gestacional (semanas)", 
       y = "Média do peso de nascimento (kg)",
       caption = "Fonte: Pacote R - GLMsData",
       title = paste0("Tentativa 03 - SQR = ", round(SC, 2)))
p03

## Visualizando as 3 tentativas lado a lado...
p01 | p02 | p03

#### Mas fiquem tranquilos, vivemos em um mundo tecnológico, temos
#### estimadores mais iterativos para realizar o processo de estimação
#### dos parâmetros...

### Utilizando a função lm - linear model...
mod0 <- lm(Weight ~ Age, weights = Births, data = db)

## visualizando os resultados do modelo...
summary(mod0)

## predizendo os resultados com base no modelo...
db$Pred <- predict(mod0, newdata = db, type = "response")

## Visualizando os valores Observados x Preditos...
p04 <- ggplot(data = db, aes(x = Weight, y = Pred)) +
  geom_point(pch = 21, fill = "white", colour = "black", size = 4) +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  coord_equal(xlim = c(0, 4), ylim = c(0, 4), expand = c(0, 0)) +
  labs(x = "Pesos observados", 
       y = "Pesos preditos",
       caption = "Fonte: Pacote R - GLMsData")
p04

## Calculando o RSS...
RSS <- sum(db$Births * (db$Weight - db$Pred)^2)
RSS

## Estimando a variância geral do modelo...

# Na mão...
gl <- nrow(db) - 2
variancia <- RSS / df
c(gl = gl, desvio.padrao = sqrt(variancia), variancia = variancia)

# Ou direto do modelo..
gl <- summary(mod0)$df[2]
variancia <- summary(mod0)$sigma^2
desvio.padrao <- summary(mod0)$sigma
c(gl = gl, desvio.padrao = desvio.padrao, variancia = variancia)

## Diagnóstico do modelo ajustado...
res <- fortify(mod0)
res$id <- 1:nrow(res)
p05 <- ggplot(data = res, aes(x = id, y = .stdresid)) +
  geom_point(pch = 21, size = 4, alpha = 0.8, fill = "white") +
  geom_hline(yintercept = 0, colour = "red") +
  labs(x = "Identificador da amostra", y = "Resíduos")
p05

############# Modelo linear Múltiplo - Y ~ X1 + X2 + ... + Xn (Todas Numéricas)

###### Carregando uma nova base de dados...
db2 <- read.table("https://www.dropbox.com/s/rgoqf210iqvf2k6/Capacidade_pulmonar.csv?dl=1",
                  header = TRUE, sep = ";", dec = ",")

###### Explorando os dados...

##### Resumo completo da base...
summary(db2)

##### A ideia aqui é avaliar a influência de diferentes fatores, tais como:
##### idade, altura, gênero e condição de tabagismo na capacidade pulmonar.
##### No entanto, neste momento, iremos avaliar a influência da idade e altura
##### dos individuos sobre a capacidade pulmonar.

##### Com isto, e considerando ainda que cada amostra em nossa base representa
##### um único individuo amostrado, temos o modelo geral estruturado da seguinte 
##### forma:

### var[y_i] = sigma2
### mu_i = beta0 + beta1 * x1 + beta2 * x2

##### Figura FEV ~ Ht...
p06 <- ggplot(data = db2, aes(x = Ht, y = FEV)) +
  geom_point(pch = 21, size = 4) +
  stat_smooth(method = "loess", formula = "y ~ x") +
  scale_y_continuous(limits = c(0, 6), expand = c(0, 0)) +
  labs(x = "Altura (Polegadas)", 
       y = "FEV (L)",
       caption = "Fonte: Pacote R - GLMsData")
p06

p07 <- ggplot(data = db2, aes(x = Ht, y = log(FEV))) +
  geom_point(pch = 21, size = 4) +
  stat_smooth(method = "loess", formula = "y ~ x") +
  scale_y_continuous(limits = c(-1, 2)) +
  labs(x = "Altura (Polegadas)", 
       y = "log(FEV (L))",
       caption = "Fonte: Pacote R - GLMsData")
p07

#### Visualizando os dois casos lado a lado...
p06 | p07

#### Modelo múltiplo - log(FEV) ~ Age + Ht...
mod1 <- lm(log(FEV) ~ Age + Ht, data = db2)

#### Visualizando os resultados do modelo...
summary(mod1)

## predizendo os resultados com base no modelo...
db2$Pred <- predict(mod1, newdata = db2, type = "response")

## Visualizando os valores Observados x Preditos...
p08 <- ggplot(data = db2, aes(x = log(FEV), y = Pred)) +
  geom_point(pch = 21, fill = "white", colour = "black", size = 4) +
  geom_abline(slope = 1, intercept = 0, color = "blue") +
  coord_equal(xlim = c(0, 2), ylim = c(0, 2), expand = c(0, 0)) +
  labs(x = "FEV observados", 
       y = "FEV preditos",
       caption = "Fonte: Pacote R - GLMsData")
p08

## Estimando a variância geral do modelo...

# Direto do modelo..
gl <- summary(mod1)$df[2]
variancia <- summary(mod1)$sigma^2
desvio.padrao <- summary(mod1)$sigma
c(gl = gl, desvio.padrao = desvio.padrao, variancia = variancia)

## Diagnóstico do modelo ajustado...
res <- fortify(mod1)
res$id <- 1:nrow(res)
p09 <- ggplot(data = res, aes(x = id, y = .stdresid)) +
  geom_point(pch = 21, size = 4, alpha = 0.8, fill = "white") +
  geom_hline(yintercept = 0, colour = "red") +
  labs(x = "Identificador da amostra", y = "Resíduos")
p09
