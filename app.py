from flask import Flask, render_template, request
import numpy as np
import joblib

modelo = joblib.load('modelo/modelo_salario.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener el valor del formulario
    experiencia = float(request.form['experiencia'])
    
    # Hacer la predicción
    prediccion = modelo.predict(np.array([[experiencia]]))
    
    return render_template('index.html', 
                           resultado=f'El salario estimado es: ${prediccion[0]:,.2f}')

if __name__ == '__main__':
    app.run(debug=True)
