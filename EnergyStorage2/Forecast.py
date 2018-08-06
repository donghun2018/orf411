"""
Forecast Object

"""
from collections import namedtuple
import numpy as np
import math
from random import randint
import numpy.linalg
import matplotlib.pyplot as plt

class Forecast():
	"""
	Base class for forecasting wind for the energy storage model
	"""

	def __init__(self, horizon, beta, sigma_x, type = 'W', forecast_L = None, a = 0.0, b = 0.0, price = 50, seed=20180529):
		"""
		Initializes the forecast
		
		:param horizon: int - length of longest forecast
		:param beta: float - beta for exponential factor of covariance matrix
		:param sigma_x: float - standard deviation of errors
		:param type: str - defines what kind of forecast it is
		:param forecast_L: Forecast - needed for a grid price forecast
		:param a: float - parameter for relation between price and demand
		:param b: float - parameter for relation between price and demand
		:param price: float - constant price of load
		:param seed: int - seed for random number generator
		"""

		self.prng = np.random.RandomState(seed)
		self.horizon = horizon
		self.beta = beta
		self.sigma_x = sigma_x
		self.current_t = 0
		self.type = type
		self.forecast_L = forecast_L
		self.a = a
		self.b = b
		self.price = price
		self.build_forecasts()

	# build new forecasts
	def build_forecasts(self):
		# wind levels
		if self.type == 'W':
			self.x_base = self.prng.uniform(low = 0.0, high = 100.0, size = self.horizon)
		# load demand level
		elif self.type == 'L':
			self.x_base = [max(0, 50 * (1 + np.cos([2 * np.pi * i / self.horizon])[0]) + 10 * self.prng.normal()) for i in range(self.horizon)]
		# grid prices
		elif self.type == 'G':
			self.build_prices()
			return
		# building prices
		elif self.type == 'P':
			self.x_base = [self.price for i in range(self.horizon)]
			self.x_forecast = [[self.price for j in range(self.horizon - i)] for i in range(self.horizon)]
			return
		# handle exception
		else:
			self.x_base = [0 for i in range(self.horizon)]
			
		# use Cholesky decomposition to generate forecasts
		cov_matrix = [[0 for i in range(self.horizon)] for j in range(self.horizon)]
		for i in range(self.horizon):
			for j in range(self.horizon):
				cov_matrix[i][j] = self.sigma_x * np.exp(- self.beta * np.absolute(i - j))
		
		CH_bar = numpy.linalg.cholesky(cov_matrix)
		self.x_forecast = [[self.x_base[i] for i in range(j, self.horizon)] for j in range(self.horizon)]
		
		for i in range(self.horizon - 1):
			f_size = i
			z = self.prng.normal(size = f_size)
			z = numpy.linalg.multi_dot([np.matrix(CH_bar)[0:f_size, 0:f_size], z])
			z = z.getA1()
			for j in range(1, len(z) + 1):
				self.x_forecast[self.horizon - i - 1][j] += z[j - 1]
				if self.x_forecast[self.horizon - i - 1][j] <= 0.0:
					self.x_forecast[self.horizon - i - 1][j] = 0.0
			if i != self.horizon - 1:
				for k in range(1, len(self.x_forecast[self.horizon - i - 1])):
					self.x_forecast[self.horizon - i - 2][k + 1] = self.x_forecast[self.horizon - i - 1][k]
		self.x_forecast[self.horizon - 1] = [self.x_base[self.horizon - 1]]
		
	# build prices forecasts
	def build_prices(self):
		self.x_base = np.array(self.forecast_L.x_base) * self.a + self.b + self.prng.normal(scale = 10.0, size = self.horizon)
		self.average = np.mean(self.x_base)
		self.x_forecast = [np.array(self.forecast_L.x_forecast[i]) * self.a + self.b + self.prng.normal(scale = 10.0, size = self.horizon - i) for i in range(self.horizon)]
		for i in range(self.horizon):
			self.x_forecast[i][0] = self.x_base[i]
			
	# return current forecast
	def return_forecast(self):
		return self.x_forecast[self.current_t]

	# increment time by 1 and reset if beyond horizon
	def t_update(self):
		self.current_t += 1
		if self.current_t > self.horizon - 1:
			self.build_forecasts()
			self.current_t = 0
		return self.current_t
		
# test module	
if __name__ == "__main__":
	
	horizon = 24
	beta = 5 * 10 ** (-2)
	sigma_x = 200
	delay = 4
	forecast = Forecast(horizon, beta, sigma_x, type = 'W', seed = np.random.randint(1000000))
	forecast_G = Forecast(horizon, beta, sigma_x, type = 'G', forecast_L = forecast, a = 1, b = 1, seed = np.random.randint(1000000))
	plt.plot(range(1, horizon + 1), forecast.x_base, c = 'black', label = 'true')
	#plt.plot(range(forecast.current_t + 1, horizon + 1), forecast.return_forecast(), label = forecast.current_t + 1)
	#plt.plot(range(forecast.current_t + 1 + delay, horizon + 1), forecast.x_forecast[delay], label = forecast.current_t + 1 + delay)
	for i in range(0, horizon, delay):
		plt.plot(range(forecast.current_t + 1, horizon + 1), forecast.return_forecast(), label = forecast.current_t + 1)
		for j in range(delay):
			forecast.t_update()
		#print(forecast.return_forecast(), forecast.t_update())
	plt.legend()
	plt.show()