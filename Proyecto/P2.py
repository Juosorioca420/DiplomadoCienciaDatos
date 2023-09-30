import pandas as pd
import numpy as np; np.random.seed(1)
import pandas_ta as pda
from datetime import date
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.stattools import adfuller

import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

import yfinance as yf

import xgboost as xgb
from keras.models import load_model

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


# DashBoard Header
# --------------------------------------------------------------------

st.title("Constructor de Portafolios Sencillo")
st.divider()

stocks = ("VEIEX", "EC") # tupla de 'base de datos'
selected_stock = st.selectbox( 'Seleccione una acción a predecir.', stocks )

n_years = st.slider( 'Horizonte a considerar: ' , 5, 25, step = 5, value = 10 )





# Load Data
# --------------------------------------------------------------------
st.header('Carga de Datos')


today = date.today().strftime("%Y-%m-%d")

temp = date.today()
start = temp - relativedelta(year= n_years)
start = start.strftime("%Y-%m-%d")

#@st.cache_data
def plot_raw(df, selected_stock):
  fig = px.line( template= 'plotly_dark' )

  fig.add_scatter(x = df['Date'], y = df['High'], name = 'Maximo', line_color = 'forestgreen')
  fig.add_scatter(x = df['Date'], y = df['Low'], name = 'Minimo', line_color = 'firebrick')
  fig.add_scatter(x = df['Date'], y = df['Adj Close'], name = 'Cierre ajustado', line_color = 'dodgerblue')

  fig.update_layout( xaxis_title ='Dia', yaxis_title ='USD$', title = f'Valor de {selected_stock} en el Tiempo',
                     xaxis_rangeslider_visible = True )
  
  return fig 


@st.cache_data # Decorador para almacenar el 'return' en memoria cache local
def load( stock = None, start = "2009-01-01", stop = today, delta_target = True, mensual = False ):
  #df = yf.download(stock, start, stop)

  if mensual:
    df = yf.download(stock, "2000-01-01", stop)

    new_indx = pd.to_datetime(df.index)
    df.set_index(new_indx, inplace=True)
    df = df.resample('M').mean()

    df['RSI'] = pda.rsi( df['Close'], length = 2 ) # indicadores de analisis tecnico
    df['FMA'] = pda.ema( df['Close'], length = 2 )
    df['MMA'] = pda.ema( df['Close'], length = 4 )
    df['SMA'] = pda.ema( df['Close'], length = 6 )

  else:
    df = yf.download(stock, start, stop)

    df['RSI'] = pda.rsi( df['Close'], length = 15 ) # indicadores de analisis tecnico
    df['FMA'] = pda.ema( df['Close'], length = 20 )
    df['MMA'] = pda.ema( df['Close'], length = 100 )
    df['SMA'] = pda.ema( df['Close'], length = 150 )

  if delta_target:
    df['Target'] = df['Adj Close'] - df['Open']
    df['Target'] = df['Target'].shift(-1) # tomorrow stock value delta
    df['Categoria'] = [ 1  if df.Target[i] > 0 else 0 for i in range( df.shape[0] ) ]
  else:
    df['Target'] = df['Adj Close'].shift(-1) # tomorrow stock value
    df['Categoria'] = [ 1  if df.Target[i] > df['Adj Close'].iloc[i] else 0 for i in range( df.shape[0] ) ]


  df.dropna(inplace=True)
  df.reset_index(inplace = True)

  df.drop( ['Volume', 'Close'], axis = 1, inplace = True )

  return df


# Add Original Data to DashBoard
data_state = st.text( 'Recolectando datos...' ) # writeo de verificacion
df = load(selected_stock,  start= start, delta_target = True, mensual = False)
data_state.text( 'Recolectando datos... Finalizado.')

st.write( df.tail() )

raw_fig = plot_raw(df, selected_stock)
st.plotly_chart(raw_fig)





# Process Data
# --------------------------------------------------------------------

samples = 0.8 # 80% de los datos
steps = 30
            # Primeras 8 columnas                                            # ultima columna
features = ['Open',	'High',	'Low', 'FMA',	'MMA', 'SMA',	'RSI', 'Adj Close']; target = ['Target']


def split(df):

  df_split = df.copy()
  df_split.drop( [ 'Date', 'Categoria' ], axis = 1, inplace = True )

  # hallar min max para transformacion inversa
  #min = df_split.iloc[steps:,-1].min(); max = df_split.iloc[steps:,-1].max()

  # normalizar datos
  scaler = MinMaxScaler(feature_range=(0,1))
  df_split = scaler.fit_transform(df_split)

  # organizar datos en steps
  x = np.array( [ df_split[ i-steps : i, 0 : 8 ] for i in range( steps, len(df_split) ) ] )
  y = np.array(df_split[steps:,-1]); y = np.reshape( y, (len(y),1) )

  #print(x.shape, y.shape)

  # train/test split
  separar = int( x.shape[0] * samples )
  x_train, x_test = x[ : separar ], x[ separar : ]
  y_train, y_test = y[ : separar ], y[ separar : ]

  return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(df)


# los datos son predichos en escala normalizada
# using scaler.inverse_transform() will fail due dimensionality
def inversa(normalizado, dataframe, borrar = True, test = True):
  '''
  Valores originales de la columna
  min, max

  Valores mínimos y máximos en el rango escalado
  0, 1
  '''
  df_inv = dataframe.copy()
  if borrar: df_inv.drop( [ 'Date', 'Categoria' ], axis = 1, inplace = True )

  if test:
    min = df_inv.iloc[steps:,-1].min()
    max = df_inv.iloc[steps:,-1].max()
  else:
    min = df_inv.iloc[:,-1].min()
    max = df_inv.iloc[:,-1].max()

  normalizado = np.ravel(normalizado)

  # Aplicar la fórmula para obtener el valor original
  original = normalizado * (max - min) + min

  return original

model_string = f'{selected_stock}.h5'
lstm = load_model(model_string)

#Prediction
y_predict = lstm.predict(x_test)
y_test = inversa(y_test, df); y_predict = inversa(y_predict, df)


mae = mean_absolute_error(y_predict, y_test)
rmse = np.sqrt( np.mean(y_predict - y_test)**2 )
mse = mean_squared_error(y_predict, y_test)


#Forecast
def forecast(model):
  '''
  Tomar los 'steps' datos más recientes para predecir recursivamente el siguiente
  '''

  temp = date.today()
  mes_atras = temp - relativedelta(months=1)
  mes_atras = mes_atras.strftime("%Y-%m-%d")


  df_forecast = df.copy()
  df_forecast.drop( [ 'Date', 'Categoria'], axis = 1, inplace = True )
  df_forecast = df_forecast.iloc[-steps:,:]

  x_df = df_forecast.iloc[ :, :8 ]

  sc = MinMaxScaler(feature_range=(0,1))
  x_df = sc.fit_transform(x_df)
  #print(df_forecast)

  x_forecast = []
  x_forecast.append(x_df) ; x_forecast = np.array(x_forecast)

  y_forecast = model.predict(x_forecast)
  y_forecast = inversa(y_forecast, df_forecast, borrar = False, test = False)


  return y_forecast[0]
y_forecast = forecast(lstm)


#Visualization
# Es necesario preservar el orden en los datos

# FIG1
split_proportion = int( df.shape[0] * 0.8 )

train = df[df.index <= split_proportion]
test = df[df.index > split_proportion]

fig1 = px.line( template= 'plotly_dark' )
fig1.add_scatter(x = train['Date'], y = train['Target'], name = 'train')
fig1.add_scatter(x = test['Date'], y = test['Target'], name = 'test')

fig1.add_shape(
    go.layout.Shape(
        type='line',
        x0= test['Date'].iloc[0], y0= df['Target'].min() - 1,
        x1= test['Date'].iloc[0], y1= df['Target'].max() + 1,
        line=dict(color='white', dash='dash'),  # Color blanco y línea punteada
    )
)
fig1.update_layout( xaxis_title ='Dia', yaxis_title ='USD$', title = f'Proporción de Datos a Entrenar')

# FIG2
x_plot = list( range( len(y_test) ) )

fig2 = px.line( template= 'plotly_dark' )
fig2.add_scatter(x = x_plot, y = (y_test), name = 'test') 
fig2.add_scatter(x = x_plot, y = (y_predict), name = 'predict')
fig2.update_layout( xaxis_title ='Muestra', yaxis_title ='USD$', title = f'Ajuste de la Curva de Diferencias')

# FIG3
diff = (y_test - y_predict) #; diff = diff[(diff >= -1) & (diff <= 1)]
fig3 = ff.create_distplot([diff], group_labels=['Diferencias'], bin_size = 0.1)

# Personaliza el diseño del gráfico
fig3.update_layout( xaxis_title='Diferencias', yaxis_title='Densidad',
                  title='Histograma de los Residuos',
                  template='plotly_dark')


st.header('Modelamiento')
st.caption('Denotando el mejor ajuste entre modelos LSTM y XGBoost')

'''
LSTM - Parametros de Entrada
  - samples : numero de registros de entrada
  - steps : numero de dias a considerar para predecir el valor siguiente
  - features : numero de series de tiempo usadas para predecir la serie target

  x_train.shape = (samples, steps, #features)
'''

result = adfuller(df['Target'])
if result[1] <= 0.05: 
  data_caption = "Los datos son estacionarios."
else: 
  data_caption = "Los datos no son estacionarios."

st.plotly_chart(fig1, use_container_width=True)
st.caption(data_caption)
st.plotly_chart(fig2, use_container_width=True)
eval_string = f'MAE: {mae:.4f} |  MSE: {mse:.4f}'
st.caption(eval_string)
st.plotly_chart(fig3, use_container_width=True)
st.caption(f'Residual Mean: {diff.mean():.4f}')
forecast_string = f'*Cambio de {selected_stock} para mañana:* {y_forecast:.3f} $USD'
st.write(forecast_string)



# Moderate-Risk Assets
st.subheader('Modelo de Markowitz')
st.caption('A septiembre de 2023 un Bono del Tesoro con riesgo AAA tiene un retorno del 4.45%')


def annualize(returns, periods = 252):
  total_periodos = returns.shape[0]
  returns = (1+returns).prod()

  return returns**(periods/total_periodos) - 1


def returns_df(original_df, name, mensual = False):
  '''
  Find the returns in time to create a Covariance Matrix
  between diferent stocks and find the Annualized return.
  '''

  if mensual:
    original_df['Date'] = pd.to_datetime(original_df['Date'])
    original_df.set_index('Date', inplace=True)

    original_df = original_df.resample('M').mean()

    original_df[name] = original_df['Adj Close'].pct_change()
    original_df.dropna(inplace = True)

  else:
    original_df[name] = original_df['Adj Close'].pct_change() # Retrono diario
    original_df.dropna(inplace = True)

  return original_df[[name]] # Retorno mensual

@st.cache_data
def load_moderate(stock, start = "2009-01-01", stop = today):

  sp500 = yf.download('VOO', start, stop); sp500.reset_index(inplace = True)
  nasdx = yf.download('VNQ', start, stop); nasdx.reset_index(inplace = True)
  stock_df = yf.download(stock, start, stop); stock_df.reset_index(inplace = True)
  lst = [sp500, nasdx, stock_df]

  d1 = returns_df(sp500, 'S&P500', True)
  d2 = returns_df(nasdx, 'VNQ', True)
  d3 = returns_df(stock_df, stock, True)

  returns_history = pd.merge(d1, d2, left_index=True, right_index=True, how='inner')
  returns_history = pd.merge(returns_history, d3, left_index=True, right_index=True, how='inner')

  return returns_history



returns_history = load_moderate(selected_stock)
st.write('Tomando los valores del SP&500 y el Vanguard Real Estate Index Fund como activos de Riesgo Moderado.')
st.write(returns_history.tail())
st.caption('Ultimos retornos mensualizados de los activos considerados.')
st.divider()


def portfolio_return(weights, returns):
  """
  Computes the return on a portfolio from constituent returns and weights

  Params.
    - weights are a Nx1 matrix
    - returns are a Nx1 matrix

  Where N is the total amount of stocks considered.
  """
  return weights.T @ returns

def portfolio_vol(weights, covmat):
  """
  Computes the volatility of a portfolio from a covariance matrix and constituent weights.

  Params.
  - Weights are a N x 1 maxtrix
  - covmat is an N x N matrix (Covariance Matrix)
  """
  return (weights.T @ covmat @ weights)**0.5


from scipy.optimize import minimize
def minimize_vol(target_return, expected_returns, cov):
  """
  Returns the optimal weights that achieve the target_return.
    Given a set of expected_returns and a covariance_matrix.

  weights : x   Optimization target parameter
  """

  # Initial values
  n = expected_returns.shape[0]
  init_guess = np.repeat(1/n, n)

  # Frontier values
  bounds = ((0, 1),) * n


  # Constraints   |   weights : x   target

  weights_sum_to_1 = {'type': 'eq', # Check when Constrain Function [fun] equals 0

                      # The funtion is 0 only when Σ(w) == 1  , So, only when x is equal to 1
                      'fun': lambda weights: np.sum(weights) - 1 }

  return_is_target = {'type': 'eq',
                      'args': (expected_returns,), # Additional parameter in 'fun' other than x

                      # So, return(x, expected_returns) equals target_return
                      'fun': lambda weights, er: target_return - portfolio_return(weights, er) }


  # Optimization Function
  weights = minimize(portfolio_vol, # Objective Function
                      init_guess, # Initial values
                      args=(cov,), # Extra arguments [ portfolio_vol(x, cov) ]
                      method='SLSQP', # Quadratic Optimizer

                      options={'disp': False},
                      constraints=(weights_sum_to_1,return_is_target),
                      bounds=bounds
                  )


  return weights.x # Return only the target parameter value


def msr(riskfree_rate, er, cov):
  """
  Returns the weights that achieve the Maximun Sharp Ratio Porfolio [MSR]

    S_p = ( promedio_retorno_portafolio - Rf ) / volatilidad_portafolio

    Por cada unidad de volatilidad, existen S_p unidades adicionales de Retorno;
    bajo un concepto de costo oportunidad en funcion del Rf

  weights : x   Optimization target parameter
  """

  # Initial values
  n = er.shape[0]
  init_guess = np.repeat(1/n, n)

  # Frontier values
  bounds = ((0, 1),) * n


  # Constraints   |   weights : x   target

  weights_sum_to_1 = {'type': 'eq', # Check when Constrain Function [fun] equals 0

                      # The funtion is 0 only when Σ(w) == 1  , So, only when x is equal to 1
                      'fun': lambda weights: np.sum(weights) - 1 }


  # Objective Function
  def neg_sharpe(weights, riskfree_rate, er, cov):
    """
    Returns the negative (so it can be minimized) of the sharpe ratio of the given portfolio
    """
    r = portfolio_return(weights, er)
    vol = portfolio_vol(weights, cov)
    return -(r - riskfree_rate)/vol


  # Optimization Function
  weights = minimize(neg_sharpe,
                      init_guess, # Initial values
                      args= (riskfree_rate, er, cov),
                      method= 'SLSQP', # Quadratic Optimizer

                      options= {'disp': False},
                      constraints= (weights_sum_to_1),
                      bounds= bounds
                  )


  return weights.x

def gmv(cov):
  """
  Returns the weights of the Global Minimum Volatility portfolio
  given a covariance matrix.
  """
  n = cov.shape[0]
  return msr(0, np.repeat(1, n), cov)

#@st.cache_data
def plot_ef(n_points = 100, expected_returns = None, cov = None, style='-', riskfree_rate= 0.0445 ):
  """
  Plots the multi-asset efficient frontier
  """

  def optimal_weights(n_points, expected_returns, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
    weights = [minimize_vol(target_return, expected_returns, cov) for target_return in target_rs]

    return weights


  weights = optimal_weights(n_points, expected_returns, cov)
  rets = [portfolio_return(w, expected_returns) for w in weights]
  vols = [portfolio_vol(w, cov) for w in weights]

  ef = pd.DataFrame({ "Returns": rets,  "Volatility": vols })

  fig, ax = plt.subplots()
  ef.plot.line(ax = ax, x="Volatility", y="Returns", style=style, legend= True, label = 'Posibles Portafolios', color= 'darkturquoise')


  #ax.set_xlim(left = 0)

  # get MSR
  w_msr = msr(riskfree_rate, expected_returns, cov)
  r_msr = portfolio_return(w_msr, expected_returns)
  vol_msr = portfolio_vol(w_msr, cov)

  # add CML
  cml_x = [0, vol_msr]
  cml_y = [riskfree_rate, r_msr]
  ax.plot(cml_x[1], cml_y[1], color='magenta', marker='*', markersize=7.5)
  ax.text( vol_msr + 0.0003, r_msr + 0.0003, 'MSR', color='magenta' )

  n = expected_returns.shape[0]

  w_ew = np.repeat(1/n, n)
  r_ew = portfolio_return(w_ew, expected_returns)
  vol_ew = portfolio_vol(w_ew, cov)

  # EW
  ax.plot([vol_ew], [r_ew], color='gold', marker='o', markersize= 5)
  ax.text( vol_ew + 0.0005, r_ew + 0.0005, 'EW', color='gold' )

  w_gmv = gmv(cov)
  r_gmv = portfolio_return(w_gmv, expected_returns)
  vol_gmv = portfolio_vol(w_gmv, cov)

  # GMW
  ax.plot([vol_gmv], [r_gmv], color='ivory', marker='o', markersize= 5)
  ax.text( vol_gmv + 0.0005, r_gmv + 0.001, 'GMV', color='ivory', fontweight='bold' )

  fig.suptitle("Frontera de Eficiencia de Markowitz")
  ax.set_ylabel("Retornos Anualizados"); ax.set_xlabel("Volatilidad")


  st.pyplot(fig)

  w_msr = np.array(w_msr).round(2); w_gmv = np.array(w_gmv).round(2)
  return [w_msr, r_msr, vol_msr], [w_gmv, r_gmv, vol_gmv]


expected_returns = annualize(returns_history, 12)
cov_matrix = returns_history.cov()
msr_port, gmv_port = plot_ef(100, expected_returns, cov_matrix)


msr_write = f'Indicadores.\n\tR = {msr_port[1]:.4f}\n\tσ = {msr_port[2]:.4f}\n\tW = {msr_port[0]}'
gmv_write = f'Indicadores.\n\tR = {gmv_port[1]:.4f}\n\tσ = {gmv_port[2]:.4f}\n\tW = {gmv_port[0]}'

st.write('**MSR**')
st.text(msr_write)
st.write('**GMV**')
st.text(gmv_write)



# CPPI
st.subheader( 'Portafolio de Proporción Constante [CPPI]' )

import ipywidgets as widgets
from IPython.display import display

def sim_cppi(risky_r, safe_returns = None, m = 3, S0 = 100, floor = 0.8, riskfree_rate = 0.0445, drawdown = None):
  """
  Run a backtest of the CPPI strategy, given a set of returns for the risky asset
  Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
  """
  # set up the CPPI parameters
  dates = risky_r.index
  n_steps = len(dates)
  account_value = S0
  floor_value = S0*floor
  peak = account_value


  # Set Safe Bonds returns by current RiskFree Rate
  if safe_returns is None:
    safe_returns = pd.DataFrame().reindex_like(risky_r)
    safe_returns.values[:] = riskfree_rate/12 # fast way to set all values to a number


  # Set DataFrames for saving intermediate values
  account_history = pd.DataFrame().reindex_like(risky_r)
  risky_w_history = pd.DataFrame().reindex_like(risky_r)
  cushion_history = pd.DataFrame().reindex_like(risky_r)
  floorval_history = pd.DataFrame().reindex_like(risky_r)
  peak_history = pd.DataFrame().reindex_like(risky_r)


  for step in range(n_steps):
    # By default, drawdown is set to fixed floor in fuction to Initial Investment S0
    # If defined, floor will grow in proportion to most recent peak
    if drawdown is not None:
      peak = np.maximum(peak, account_value)
      floor_value = peak*(1-drawdown)

    cushion = (account_value - floor_value)/account_value

    risky_w = m*cushion
    risky_w = np.minimum(risky_w, 1)
    risky_w = np.maximum(risky_w, 0)

    safe_w = 1-risky_w

    risky_alloc = account_value*risky_w
    safe_alloc = account_value*safe_w

    # recompute the new account value at the end of this step
    account_value = risky_alloc*( 1 + risky_r.iloc[step] ) + safe_alloc*( 1 + safe_returns.iloc[step] )

    # save the histories for analysis and plotting
    cushion_history.iloc[step] = cushion
    risky_w_history.iloc[step] = risky_w
    account_history.iloc[step] = account_value
    floorval_history.iloc[step] = floor_value
    peak_history.iloc[step] = peak

  risky_wealth = S0*(1+risky_r).cumprod()


  simulation = {"Wealth": account_history,
                "Risky Wealth": risky_wealth,
                "Risk Budget": cushion_history,
                "Risky Allocation": risky_w_history,
                "m": m,
                "S0": S0,
                "floor": floor,
                "risky_r":risky_r,
                "safe_returns": safe_returns,
                "drawdown": drawdown,
                "peak": peak_history,
                "floor": floorval_history
                }

  return simulation


def brownian_motion(n_years = 10, n_scenarios = 100, mu = 0.07, sigma = 0.15, steps_per_year = 12, S0 = 100.0, en_dolares = False):
  """
  Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices.

  Parameters:
    years:  The number of years to generate data for
    scenarios: The number of different scenarios/trajectories
    mu: Annualized Drift, i.e Market Return
    sigma: Annualized Volatility
    steps_per_year: granularity of the simulation; periods per year
    S_0: initial value

  Return:
    Columns amount of scenarios
    Rows years * steps_per_year
  """
  np.random.seed(1)

  dt = 1/steps_per_year ;  n_steps = int(n_years*steps_per_year) + 1
                # data distribution    drift           volatility
  rets_plus_1 = np.random.normal(loc= ( 1 + mu )**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps + 1, n_scenarios))
  rets_plus_1[0] = 1

                                                              # Raw Returns
  res = S0*pd.DataFrame(rets_plus_1).cumprod() if en_dolares else (rets_plus_1 - 1)
  return res


#@st.cache_data
def plot_cppi(n_scenarios = 50, mu = 0.12, sigma = 0.02, m = 3, floor = 0.8, riskfree_rate = 0.0445, y_max = 100, steps_per_year = 12):
  """
  Plot the results of a Monte Carlo Simulation of CPPI

    - n : total scenarios
    - mu : weighted avg. of portfolio returns
    - sigma : weighted volatility of portfolio
    - m : Risky asset multiplier
    - floor : maximun acetable proportion of losses by initial investment [S0]
    - R_f : risk free rate (Treasury Bond)
    - y_max : plot y axis scale
    - steps_per_year : CPPI portfolio actualization frequency. ( default: monthly )
  """
  S0  = 100 # Nominal Inicial
  sim_rets = brownian_motion(n_scenarios = n_scenarios, mu = mu, sigma = sigma, steps_per_year = steps_per_year)
  risky_r = pd.DataFrame(sim_rets) # Retornos generados con mov. Borowniano



  # Run the back-test
                           # working with Series may prove to be inconvinient
  sim = sim_cppi(risky_r = pd.DataFrame(risky_r), riskfree_rate = riskfree_rate, m = m, S0 = S0 , floor=floor)
  wealth = sim["Wealth"]
  y_max = wealth.values.max()*y_max/100 # y axis Zoom



  # Calculate terminal wealth stats

  terminal_wealth = wealth.iloc[-1]

  tw_mean = terminal_wealth.mean()  # ret. promedio
  tw_median = terminal_wealth.median() # ret. medio

  fail = np.less(terminal_wealth, S0*floor)
  n_failures = fail.sum()
  p_fail = n_failures/n_scenarios

  avg_shortfall = np.dot( terminal_wealth - S0 * floor,  fail ) / n_failures if (n_failures > 0) else 0


  # Plot

  fig, (wealth_ax, hist_ax) = plt.subplots(nrows = 1, ncols = 2, sharey = True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
  plt.subplots_adjust(wspace=0.0) # Adjacent plots

  wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="darkturquoise")
  wealth_ax.axhline(y= S0, ls=":", color="ivory")
  wealth_ax.axhline(y= S0*floor, ls="--", color="red", linewidth=2)
  wealth_ax.set_ylim(top=y_max)


  terminal_wealth.plot.hist(ax = hist_ax, bins=50, ec='darkturquoise', fc='darkturquoise', orientation='horizontal', alpha = 0.8)
  hist_ax.axhline(y= S0, ls=":", color="ivory")
  hist_ax.axhline(y= tw_mean, ls="--", color="gold")
  hist_ax.axhline(y= tw_median, ls="--", color="magenta")


  hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(0.75, 0.99),xycoords='axes fraction', fontsize= 14, color = 'gold')
  hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(0.75, 0.95),xycoords='axes fraction', fontsize= 14, color = 'magenta')

  if (floor > 0.01):
    hist_ax.axhline(y=S0*floor, ls="--", color="red", linewidth=1)
    hist_ax.annotate(f'Violations: {n_failures} [{p_fail*100:2.2f}%]\nE(shortfall): ${avg_shortfall:2.2f}', xy=(0.75, 0.85), xycoords='axes fraction', fontsize= 14)

  return fig


# Definir los controles interactivos en Streamlit
col1, col2, col3 = st.columns(3)

n_scenarios = st.slider("Número de Escenarios", min_value=1, max_value=500, step=5, value=50)

with col1:
  mu = st.slider("Mu", min_value=-0.2, max_value=0.2, step=0.01, value=0.11)
  floor = st.slider("Floor", min_value=0.0, max_value=2.0, step=0.1, value=0.8)

with col2:
  sigma = st.slider("Sigma", min_value=0.0, max_value=0.2, step=0.01, value=0.03)
  M = st.slider("M", min_value=1.0, max_value=5.0, step=0.5, value=3.0)

with col3:
  riskfree_rate = st.slider("Tasa Libre de Riesgo", min_value=0.0, max_value=0.05, step=0.01, value=0.04)
  steps_per_year = st.slider("Períodos por Año", min_value=1, max_value=12, step=1, value=12)


cppi_plot = plot_cppi(n_scenarios = n_scenarios, mu = mu, sigma = sigma, 
                      m = M, floor = floor, 
                      riskfree_rate = riskfree_rate, 
                      steps_per_year = steps_per_year)
st.pyplot(cppi_plot)