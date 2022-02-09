# Previsão de Vendas

Qual empresa nunca se viu na necessidade de estimar o seu volume de vendas futuro? 

A previsão desse tipo de variável é muito comum no mundo dos negócios, pois é essa expectativa que direciona as grandes decisões de uma empresa.

## Entendimento do Problema
A Rossmann possui mais de 3.000 drogarias em 7 países na Europa. Atualmente, os gerentes de cada loja têm a tarefa de prever suas vendas diárias com seis semanas de antecedência. As vendas da loja são influenciadas por muitos fatores, desde promoções até distância de competidores. Com milhares de gerentes individuais prevendo vendas, a precisão dos resultados pode ser bastante variada. 

**O grande problema, então, é o fato das previsões terem uma precisão muito variada, não trazendo tanta segurança ao negócio.**

## Solução
Para solucionar o problema de negócio se faz necessário o uso de **algoritmos de Machine Learning (Time Series Regression)**

Esse algoritmo deverá ser colocado **em produção na nuvem através de uma API**. 

**Tecnologias Utilizadas:** 
* Python, Git e Github
* Bibliotecas de manipulação de dados (Pandas, Numpy..)
* Bibliotecas de visualização de dados (Seaborn e Matplotlib)
* Bibliotecas de machine learning (Sklearn, XGBoost...)
* Bibliotecas para deploy e criação da API (Flask e Pickle)
* Heroku para hospedar nossa API na cloud.

## Data Description

Os dados são compostos por 1115 lojas em um período de 2013 até agosto de 2015 com diversas variáveis, inclusive o target: 'Sales'.
 
```
Store - ID de cada loja
 
Sales - Quantidade de vendas feitas por dia (Target Variable)
 
Customers - Quantidade de clientes que foram à loja naquele dia.
 
Open - Indica se a loja está fechada ou não (Open == 1, Closed == 0)
 
StateHoliday - Indica se foi um dia de feriado estadual (a = Feriado Público, b = Páscoa, c = Natal, 0 = Sem Feriado)
 
SchoolHoliday - Indica se foi um dia de feriado escolar. (Holiday == 1, No Holiday == 0)
 
StoreType  - São os tipos de loja, existem 4 (a, b, c, d)
 
Assortment - Indica o nível de sortimento da loja (a = Básico, b = Extra, c = Estendido)
 
CompetitionDistance - Distância, em metros, do concorrente mais próximo
 
CompetitionOpenSince[ Month / Year ] - Fornece o mês e o ano, em duas variáveis distintas, que o competidor dessa loja abriu.
 
Promo - Indice se há promoção na loja naquele dia (Promo == 1, No promo == 0)
 
Promo2 - Promo2 é uma promoção contínua e consecutiva (Loja não está participando == 1, Loja não está participando == 0)
 
Promo2Since[ Week / Year ] - Fornece a semana e o ano, em duas variáveis distintas, que a loja passou a aderir à Promo2.
 
PromoInterval - Descreve os intervalos em que a Promo2 é iniciada, nomeando os meses em que a promoção é reiniciada. Por exemplo, "fevereiro, maio, agosto, novembro" significa que a Promo2 começa em fevereiro, maio, agosto, novembro de qualquer ano para aquela loja
```

## Model Performance

#### Modelo Baseline
 
|       Model Name          |        MAE          |      MAPE      |        RMSE        |
|:-------------------------:|:-------------------:|:--------------:|:------------------:|
| Baseline                  |  1440          | 22.8%      |       1925        |

#### Tuned XGBoost Regressor (Our model)

|       Model Name          |        MAE          |      MAPE      |        RMSE        |
|:-------------------------:|:-------------------:|:--------------:|:------------------:|
| Baseline                  |  979          | 14.6%      |       1413        |

Vemos uma precisão claramente maior, **um erro médio com diminuição de 32%!**

Agora a empresa pode ter muito mais segurança nas suas expectativas futuras de faturamento!

Esse modelo está disponível na cloud, podendo ser acessado <a href=https://reqbin.com/oqcbaldp target="_blank">aqui</a>
(É só mudar as variáveis na aba 'Content')

## Insights

Além de uma maior precisão nas previsões, **esse projeto conseguiu gerar insights de grande valor para o negócio**. Vou listar apenas três dos diversos observados ao longo do projeto.

**1º Insight: As lojas que acataram a Promo2 vendem MENOS do que as que não acataram!**

Isso é muito contra intuitivo, pois geralmente pensamos que promoções atraem mais vendas! Isso é um insight especialmente importante para o negócio, pois a partir dele a empresa pode rever o modelo de negócio vigente!

Abaixo está a análise do volume de vendas relativo entre as duas situações.

<img src="https://i.imgur.com/x9smoqv.png"/>

**2º Insight: Lojas com competidores distantes NÃO necessariamente vendem mais**

Tendemos a pensar que isso seria certamente uma verdade, pois quanto mais competição há no mercado, menos cada empresa venderia. Apesar disso, o que foi observado é uma tendência de correlação quase nula. Então, a Rossmann pode abrir drogarias em locais com mais competidores, pois isso não impactará o volume de vendas dessa loja.

Abaixo está a análise da correlação entre volume de vendas e distância de competidores.

<img src="https://i.imgur.com/99Q8D75.png"/>

**3º Insight: O volume de vendas aos finais de semana é MENOR do que os dos dias de semana, mas os consumidores consomem basicamente a mesma coisa!**

O que acontece nos fins de semana é que há um menor fluxo de pessoas, o que acarreta em menores vendas! A Rossmann pode tentar atrair mais consumidores através de promoções e outros artifícios, o que ajudaria a manter o volume de faturamento diário (OBS: Considero, aqui, fim de semana sendo o sábado, pois o domingo é em geral de lojas fechadas)

Abaixo está a análise do volume de vendas relativo entre dias de semana e fins de semana.

<img src="https://i.imgur.com/M0RjwqZ.png"/>

Abaixo está a análise do volume de vendas por cada consumidor entre dias de semana e fins de semana.

<img src="https://i.imgur.com/9acF04K.png"/>

## Conclusão

Esse projeto nos **rendeu diversos insights valiosos para o negócio, que vão possibilitar a Rossmann tomar decisões mais acertadas** para aumentar seu faturamento, e **possibilitou que a empresa conseguisse ter mais segurança na previsão de vendas, com uma diminuição de 32% no erro médio** comparado ao modelo base.

Ainda há muito que pode ser feito: Coletar mais dados, formular novas hipóteses, gerar novos insights na EDA, testar outros data transformers, testar outros algoritmos, utilizar um Hyperparameter Fine Tuning exaustivo como o GridSearchCV, dentre outras milhões de possibilidades! 

Apesar desse mar de coisas novas para fazer, considero que o que temos atualmente supre a necessidade da empresa no curto prazo. Dessa forma **entregamos valor de forma rápida** e vamos aperfeiçoando o projeto até onde for necessário para o negócio.
