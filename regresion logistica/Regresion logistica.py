# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated", category=FutureWarning)

datos = pd.read_csv('D:/bsaan/OneDrive - Universidad Nacional Abierta y a Distancia/universidad/Septimo semestre/Analisis de datos/actividad 3/solucion/regresion logistica/heart_cleveland_upload - original.csv')

# Visualizar las primeras filas del dataset
print(datos.head())


# Obtener información sobre el dataset
print(datos.info())

# Resumen estadístico de los datos
print(datos.describe())

# Identificar valores faltantes
print(datos.isnull().sum())

# Eliminar filas con valores faltantesos
datos = datos.dropna()

# Visualizar relaciones entre variables
sns.pairplot(datos)
plt.show()

# Paso 3: Selección de características
# Seleccionar características relevantes
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'ca']
X = datos[features]
y = datos['condition']

# Paso 4: División del dataset en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=1000)  # Aumenta el número máximo de iteraciones
model.fit(X_train, y_train)

# Paso 5: # Predecir en el conjunto de prueba
y_pred = model.predict(X_test)


# Paso 6: Evaluar el modelo, Evaluar el desempeño del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)



# Paso 7: Visualización de resultados
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
