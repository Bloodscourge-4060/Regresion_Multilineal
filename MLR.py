# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt


# data = pd.read_csv('Datos.csv')

# # Separar las variables independientes (X1 y X2) y la variable dependiente (Y)
# X = data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
# y = data['Performance Index']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# model = LinearRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)


# r2 = r2_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print("Coeficiente de determinación (R^2):", r2)
# print("Error cuadrático medio (MSE):", mse)

# plt.scatter(y_test, y_pred)
# plt.xlabel("Valores Reales")
# plt.ylabel("Predicciones")
# plt.title("Valores Reales vs. Predicciones")
# plt.show()


# residuals = y_test - y_pred
# plt.scatter(y_test, residuals)
# plt.xlabel("Valores Reales")
# plt.ylabel("Errores (Residuales)")
# plt.title("Errores vs. Valores Reales")
# plt.axhline(y=0, color='r', linestyle='--')
# plt.show()

# plt.hist(residuals, bins=30)
# plt.xlabel("Errores (Residuales)")
# plt.ylabel("Frecuencia")
# plt.title("Distribución de Errores (Residuales)")
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Datos.csv')

X = data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = data['Performance Index']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

b0 = model.intercept_ #Esta es la ordenada al origen
b1, b2, b3, b4 = model.coef_ #Es para los coeficientes de las variables independientes

# Aqui se calcula la línea de regresión
linea_regresion = b0 + b1 * X['Hours Studied'] + b2 * X['Previous Scores'] + b3 * X['Sleep Hours'] + b4 * X['Sample Question Papers Practiced']

plt.scatter(y, y_pred)
plt.plot(X['Previous Scores'], linea_regresion, color='green')
plt.legend(['Valores reales vs Valores Predichos', 'Previous Scores'])
plt.xlabel("Performance Index")
plt.ylabel("Performance Index Prediction")
plt.show()
