import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
# Ignorar advertencias
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Load the dataset
df = pd.read_csv('D:/bsaan/OneDrive - Universidad Nacional Abierta y a Distancia/universidad/Septimo semestre/Analisis de datos/Actividad 5/Titanic-Dataset.csv')
# Mostrar las primeras filas del marco de datos
print("Visualización primaria del dataSet")
print(df.head())
print("Tipos de datos de cada columna:")
print(df.dtypes)
# Corroborando cuantos valores son NULL por columna
print("Valores NULL de cada columna del DataFrame.")
print(df.isnull().sum())

# Impute missing values
df['Edad'].fillna(df['Edad'].median(), inplace=True)
df['Embarcado'].fillna(df['Embarcado'].mode()[0], inplace=True)
df['Cabina'].fillna(df['Cabina'].mode()[0], inplace=True)
# Casteando valores String a numericos para procesamiento necesario
df['Sexo'] = df['Sexo'].replace({'male': 1, 'female': 0})
# Quitando columnas irrelevantes
df.drop(['IdPasajero', 'Nombre', 'Ticket', 'Cabina'], axis=1, inplace=True)
# Estadísticas de resumen de variables numéricas
print(df.describe())
# Obtener los valores únicos de la columna 'Embarcado' porque necesito convertir la columna a tipo numérico
print("Valores unicos de Embarcado:")
print(df['Embarcado'].unique())

# S = 0, C = 1, Q = 2
df['Embarcado'] = df['Embarcado'].replace({'S': 0, 'C': 1, 'Q': 2})
# Correlation matrix
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

"""
La variable más importante es "Sobrevivió", que indica si una persona sobrevivió o no.
Tiene una correlación negativa moderada con "Clase" (-0.34), lo que sugiere que las personas en clases más altas tenían más probabilidades de sobrevivir.
Tiene una correlación negativa fuerte con "Sexo" (-0.54), indicando que el sexo tiene una influencia significativa en la supervivencia (posiblemente mujeres tuvieron más probabilidades de sobrevivir).
Correlaciones muy débiles con otras variables como "Edad" (-0.065), "Hermanos/Cónyuge" (-0.035), "Padres/Hijos" (0.082), "Tarifa" (0.26), y "Embarcado" (0.11).
"""

# Data Visualization
sns.pairplot(df, hue='Sobrevivió')
plt.show()

"""
Mirando todos los diagramas saque un top 5 para orientarme en el análisis exploratorio:

Sexo vs. Sobrevivió:
Este gráfico es crucial porque la variable de sexo ha demostrado ser una de las más significativas en la tasa de supervivencia del Titanic.

Clase vs. Sobrevivió:
La clase del pasajero (1ª, 2ª, 3ª) también es un factor crítico. Los pasajeros de clases superiores tenían mayores tasas de supervivencia.

Edad vs. Sobrevivió:
La edad es una variable importante. Por ejemplo, se ha observado que los niños tenían una mayor probabilidad de supervivencia.

Tarifa vs. Sobrevivió:
La tarifa pagada por el billete puede estar correlacionada con la clase y, por lo tanto, con la probabilidad de supervivencia. Es útil explorar esta relación directamente.

Embarcado vs. Sobrevivió:
El puerto de embarque (C = Cherburgo, Q = Queenstown, S = Southampton) puede proporcionar información sobre la distribución de pasajeros y su supervivencia.
"""

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Clase', y='Edad', data=df)
plt.title('Diagrama de caja de Edad por Clase')
plt.show()

# Feature Engineering
# Convert categorical variables to dummy variables
#df = pd.get_dummies(df, columns=['Sexo', 'Embarcado'], drop_first=True)
# Construcción de modelos (regresión)

# Columnas relevantes
columnas_relevantes = ['Clase', 'Sexo', 'Edad', 'Hermanos/Cónyuge', 'Padres/Hijos', 'Tarifa', 'Embarcado']

# Dividir el conjunto de datos en características y variable de destino
X = df[columnas_relevantes]
y = df['Sobrevivió']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_relevantes),
    ])

# Definir el modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Predecir y evaluar
y_pred = model.predict(X_test)
print('Metricas de evaluacion del modelo en el conjunto de test:')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-score: {f1_score(y_test, y_pred)}')

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de confusión:')
print(conf_matrix)
# Supongamos que tienes un DataFrame llamado 'data' que contiene tus datos
# Asegúrate de tener la columna 'Sobrevivió' en tus datos

# Diagrama de caja para Clase y Sobrevivió
sns.boxplot(x='Sobrevivió', y='Clase', data=df)
plt.title('Diagrama de Caja para Clase y Sobrevivió')
plt.show()

# Diagrama de caja para Sexo y Sobrevivió
sns.boxplot(x='Sobrevivió', y='Sexo', data=df)
plt.title('Diagrama de Caja para Sexo y Sobrevivió')
plt.show()

# Diagrama de caja para Edad y Sobrevivió
sns.boxplot(x='Sobrevivió', y='Edad', data=df)
plt.title('Diagrama de Caja para Edad y Sobrevivió')
plt.show()
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Obtener los coeficientes del modelo de regresión logística
coefficients = model.named_steps['classifier'].coef_[0]
features = columnas_relevantes

plt.figure(figsize=(12, 6))
sns.barplot(x=coefficients, y=features)
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients')
plt.show()
