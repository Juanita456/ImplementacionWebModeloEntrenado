from flask import Flask, render_template, request
import numpy as np
import joblib
# Cargar el modelo entrenado
modelo = joblib.load('modelo/modelo_regresion_logistica.pkl')
# Crear la app Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener los datos del formulario
    edad = float(request.form['edad'])
    salario = float(request.form['salario'])
    # Convertir a arreglo 2D (ya que el modelo espera una matriz)
    entrada = np.array([[edad, salario]])
    # Hacer la predicción
    prediccion = modelo.predict(entrada)
    probabilidad = modelo.predict_proba(entrada)[0][1] * 100  # % de probabilidad de compra
    # Interpretar el resultado
    if prediccion[0] == 1:
        resultado = f"El usuario probablemente COMPRARÁ el producto (confianza: {probabilidad:.2f}%)"
    else:
        resultado = f"El usuario probablemente NO comprará el producto (confianza: {probabilidad:.2f}%)"
    
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
