import numpy as np
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog

# Función para leer un archivo csv y guardar su contenido en tres matrices
def guardar_Csv(filename):
    Matriz = []
    UltimosValores = []
    OtrosValores = []

    with open(filename, 'r') as file:
        lector = csv.reader(file)

        for fila in lector:
           if fila:
                # Dividir la cadena en partes usando el punto y coma y convertir cada parte a un número
                fila_numeros = [float(valor.replace(',', '.')) for valor in fila[0].split(';')]
                Matriz.append(fila_numeros)

                # Guardar el último valor de la fila en la matriz de últimos valores
                UltimosValores.append(fila_numeros[-1])

                # Guardar todos los valores excepto el último en la matriz de otros valores
                OtrosValores.append(fila_numeros[:-1])


    # Convertir las listas a matrices NumPy
    matriz_np = np.array(Matriz)
    Salida_np = np.array(UltimosValores)
    Entradas_np = np.array(OtrosValores)

    return matriz_np, Salida_np, Entradas_np

numeroDeEntradas = 0
tasa_de_aprendizaje = 0.001
epocas = 100
norma_error=[]
historial_pesos=[]
# Función para inicializar los pesos y el sesgo
def inicializar_pesos():
    pesos = np.random.uniform(-1, 1, size=numeroDeEntradas)
    sesgo = np.random.uniform(-1, 1)
    return pesos, sesgo

# Función de activación corregida
def activacion(z, b):
    if z + b > 0:
        return 1
    else:
        return 0

# Función de entrenamiento corregida
def entrenamiento(epocas, entradas, salida):
    global numeroDeEntradas, norma_error,historial_pesos  # Agregamos esta línea para acceder a la variable global
    numeroDeEntradas = len(entradas[0])

    pesos, sesgo = inicializar_pesos()

    for epoca in range(epocas):
        error_total = 0

        for i in range(len(entradas)):
            # Calcular la suma ponderada
            suma_ponderada = np.dot(pesos, entradas[i]) + sesgo
            
            # Aplicar la función de activación
            prediccion = activacion(suma_ponderada, sesgo)

            # Calcular el error
            error = salida[i] - prediccion
            error_total += error**2

            # Actualizar pesos y sesgo utilizando la regla de aprendizaje del perceptrón
            pesos += tasa_de_aprendizaje * entradas[i] * error
            sesgo += tasa_de_aprendizaje * error


        norma_error.append(error_total)
        historial_pesos.append(np.copy(pesos)) 
        print(f"Época {epoca + 1}, Error total: {error_total}")
    print(historial_pesos)
    
    print(f"Los pesos finales son: {pesos}")

        # Después de realizar el entrenamiento (dentro de la función entrenamiento)
    tasa_aprendizaje_label.config(text=f"Tasa de Aprendizaje: {tasa_de_aprendizaje}")
    error_permisible_label.config(text="Error Permisible: 0")
    iteraciones_label.config(text=f"Cantidad de Iteraciones: {epocas}")
    pesos_iniciales_label.config(text=f"Pesos Iniciales: {inicializar_pesos()[0]}")
    pesos_finales_label.config(text=f"Pesos Finales: {pesos}")

    return pesos, sesgo

def leer_csv(ruta):
    try:
        with open(ruta, 'r') as archivo_csv:
            lector_csv = csv.reader(archivo_csv)
            
            # Iterar sobre las filas del archivo CSV e imprimir cada fila
            for fila in lector_csv:
                print(fila)
    except FileNotFoundError:
        print(f"El archivo {ruta} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

def mostrar_resultados(resultado):
    resultado_label.config(text=resultado)

def graficas_entrenamiento(norma_error, historial_pesos):
    # Crear la subgráfica para la Norma del Error
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(norma_error) + 1), norma_error)
    plt.xlabel('Época')
    plt.ylabel('Norma del Error')
    plt.title('Evolución de la Norma del Error por Época')

    # Crear la subgráfica para la Evolución de las Posiciones
    plt.subplot(1, 2, 2)
    pesos = np.array(historial_pesos)
    for i in range(pesos.shape[1]):
        plt.plot(range(1, len(historial_pesos) + 1), pesos[:, i], label=f'Posición {i + 1}')

    plt.title('Evolución de las Posiciones por Época')
    plt.xlabel('Época')
    plt.ylabel('Valor de la Posición')
    plt.legend()
    plt.grid(True)

    # Ajustar diseño y mostrar la figura
    plt.tight_layout()
    plt.show()
# Función para obtener la tasa de aprendizaje y el número de épocas desde la interfaz
def obtener_parametros():
    global tasa_de_aprendizaje, epocas
    tasa_de_aprendizaje = float(entry_tasa_aprendizaje.get())
    epocas = int(entry_epocas.get())


def seleccionar_archivo():
    obtener_parametros() 
    ruta_archivo = filedialog.askopenfilename(
        title="Seleccionar archivo CSV",
        filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
    )
    
    if ruta_archivo:
        matriz_leida, salida, entradas = guardar_Csv(ruta_archivo)
        pesos, sesgo = entrenamiento(epocas, entradas, salida)
        mostrar_resultados(f"Los pesos finales son: {pesos}")
        graficas_entrenamiento(norma_error, historial_pesos)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Lector de CSV")

# Crear cuadros de entrada para la tasa de aprendizaje y el número de épocas
label_tasa_aprendizaje = tk.Label(ventana, text="Tasa de Aprendizaje:")
label_tasa_aprendizaje.pack()
entry_tasa_aprendizaje = tk.Entry(ventana)
entry_tasa_aprendizaje.pack()

label_epocas = tk.Label(ventana, text="Número de Épocas:")
label_epocas.pack()
entry_epocas = tk.Entry(ventana)
entry_epocas.pack()

# Crear un botón para seleccionar el archivo
boton_seleccionar = tk.Button(ventana, text="Seleccionar Archivo CSV", command=seleccionar_archivo)
boton_seleccionar.pack(pady=20)

# Crear widgets para mostrar resultados en la interfaz gráfica
resultado_label = tk.Label(ventana, text="")
resultado_label.pack()

# Etiquetas para mostrar configuración de pesos
pesos_iniciales_label = tk.Label(ventana, text="")
pesos_finales_label = tk.Label(ventana, text="")
pesos_iniciales_label.pack()
pesos_finales_label.pack()

# Etiquetas para mostrar tasa de aprendizaje, error permisible y cantidad de iteraciones
tasa_aprendizaje_label = tk.Label(ventana, text=f"Tasa de Aprendizaje: {tasa_de_aprendizaje}")
error_permisible_label = tk.Label(ventana, text="Error Permisible: 0")
iteraciones_label = tk.Label(ventana, text=f"Cantidad de Iteraciones: {epocas}")

tasa_aprendizaje_label.pack()
error_permisible_label.pack()
iteraciones_label.pack()

# Iniciar el bucle de la interfaz gráfica
ventana.mainloop()


"""# Prueba del perceptrón
prueba = True
while prueba==True:
    #pide entrada de datos
    entrada = input("Ingrese los datos separados por coma: ")
    #separar los datos por coma
    datos = entrada.split(",")
    #convertir los datos a flotantes
    datos = [float(dato) for dato in datos]
    #calcular la suma ponderada
    suma_ponderada = np.dot(pesos, datos) + sesgo
    #aplicar la función de activación
    prediccion = activacion(suma_ponderada, sesgo)
    #imprimir la predicción
    print(f"La predicción es: {prediccion}")
    #preguntar si quiere salir
    entrada = input("Si quiere salir escriba 'salir': ")
    

    #if salir sale
    if entrada == "salir":
        prueba = False"""

"""numeroDeEntradas = len(entradas[0])
tasa_de_aprendizaje = 0.01
epocas = 100


def generarPesos():
    return np.random.uniform(-1, 1, size=numeroDeEntradas)
    
def generarBias():
    return np.random.uniform(-1, 1)

def sumatoria(pesos,x):
    z = pesos * x
    return z

def activacion( z,b):


    if z.sum() + b > 0:
        return 1
    else:
        return 0



def entrenamiento(epocas, entradas, salida):
    pb = generarBias()
    px = generarPesos()

    for epoca in range(epocas):
        error_total = 0

        for i in range(len(entradas)):
            # Calcular la suma ponderada
            suma_ponderada = sumatoria(px, entradas[i])
            
            # Aplicar la función de activación
            prediccion = activacion(suma_ponderada, pb)

            error = salida[i] - prediccion
            error_total += error**2
            pesos[0] += tasa_de_aprendizaje * personas[i][0] * error
            pesos[1] += tasa_de_aprendizaje * personas[i][1] * error
            b += tasa_de_aprendizaje * error 
        print(error_total, end=" ")"""
















"""""""""

# w1*x1 + w2*x2 + ⋯ + wn*xn

def activacion(pesos, x, b):
    z = pesos * x
    if z.sum() + b > 0:
        return 1
    else:
        return 0

pesos = np.random.uniform(-1, 1, size=2)
b = np.random.uniform(-1, 1)


activacion(pesos, [0.5, 0.4], b)


pesos = np.random.uniform(-1, 1, size=2)
b = np.random.uniform(-1, 1)
tasa_de_aprendizaje = 0.01
epocas = 100

for epoca in range(epocas):
    error_total = 0
    for i in range(len(personas)):
        prediccion = activacion(pesos, personas[i], b)
        error = clases[i] - prediccion
        error_total += error**2
        pesos[0] += tasa_de_aprendizaje * personas[i][0] * error
        pesos[1] += tasa_de_aprendizaje * personas[i][1] * error
        b += tasa_de_aprendizaje * error 
    print(error_total, end=" ")

activacion(pesos, [0.5, 1], b)"""""""""