# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated", category=FutureWarning)

datos = pd.read_csv('D:/bsaan/OneDrive - Universidad Nacional Abierta y a Distancia/universidad/Septimo semestre/Analisis de datos/actividad 3/solucion/regresion logistica/heart_cleveland_upload - original.csv')

# Visualizar las primeras filas del dataset
print(datos.head())

# Obtener información sobre el dataset
print(datos.info())

# Resumen estadístico de los datos
print(datos.describe())

# Paso 2: Preprocesamiento de datos
# Identificar valores faltantes
print(datos.isnull().sum())

# Dado que no se encontraron valores faltantes en los datos, no es necesario realizar imputación de datos.
# La eliminación de filas con valores faltantes (datos = datos.dropna()) no es aplicable en este caso.

print(datos.dtypes)

# Calcular la media y la desviación estándar de los datos
mean = datos.mean()
std = datos.std()

# Calcular los límites de los datos atípicos
outlier_threshold = 3
lower_bound = mean - outlier_threshold * std
upper_bound = mean + outlier_threshold * std

# Encontrar los índices de las filas que contienen datos atípicos
outlier_indices = []
for column in datos.columns:
    outlier_indices.extend(datos[(datos[column] < lower_bound[column]) | (datos[column] > upper_bound[column])].index)

# Eliminar los índices duplicados
outlier_indices = list(set(outlier_indices))

# Crear un nuevo DataFrame con los datos atípicos
outliers = datos.loc[outlier_indices]

# Visualizar los datos atípicos
print("Datos atípicos:")
print(outliers)

# Visualizar relaciones entre variables
sns.pairplot(datos)
plt.show()

# Paso 3: Seleccionar características relevantes
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'ca']
X = datos[features]
y = datos['condition']

# Paso 4: División del dataset en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression(max_iter=1000)  # Aumenta el número máximo de iteraciones
model.fit(X_train, y_train)

# Paso 5: Predecir en el conjunto de prueba, calcular probabilidades de predicción en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Paso 6: Evaluar el modelo, Evaluar el desempeño del modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Exactitud:", accuracy)
print("Matrix de Confusion :\n", conf_matrix)
print("Reporte de Classificación:\n", class_report)

# Paso 7: Realizar las diferentes gráficas que permitan visualizar los resultados del modelo.

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Graficar la distribución de las predicciones
plt.figure(figsize=(8, 6))
sns.histplot(y_pred, kde=False, color='skyblue')
plt.xlabel('Predicted Condition')
plt.ylabel('Count')
plt.title('Predicted Condition Distribution')
plt.show()

# Graficar la distribución de las edades de los pacientes en el conjunto de prueba
plt.figure(figsize=(8, 6))
sns.histplot(datos.loc[y_test.index, 'age'], kde=True, color='salmon')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribución de las edades en el dataSet de test')
plt.show()

print("Accuracy:", accuracy)
print("Classification Report:\n", class_report)

# de la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calcular el área bajo la curva ROC (AUC)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('Falsos negativos Rate')
plt.ylabel('Verdaderos positivos')
plt.title('Curva de característica operativa del receptor (ROC)')
plt.legend(loc='lower right')
plt.show()

# Imprimir el área bajo la curva ROC (AUC)
print("AUC Score:", auc)
