import tensorflow as tf
import numpy as np
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_PATH = 'modelo_vehiculos.keras'

# Nombres de las clases (deben coincidir con el dataset de entrenamiento)
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']


# ==========================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================
def predecir_imagen(model, image_path):
    try:
        # 1. Cargar imagen
        img = Image.open(image_path)
        # Convertir a RGB por si la imagen es PNG con transparencia
        img = img.convert('RGB')

        # 2. Redimensionar a 32x32 (tamaño que espera el modelo)
        img = img.resize((32, 32))

        # 3. Preparar datos para el modelo
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalizar
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de lote

        # 4. Predecir
        predictions = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(predictions[0])

        nombre_clase = class_names[np.argmax(score)]
        confianza = 100 * np.max(score)

        return nombre_clase, confianza

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None, None


# ==========================================
# INTERFAZ GRÁFICA PARA SELECCIONAR ARCHIVO
# ==========================================
def seleccionar_archivo_gui():
    # Crear una ventana raíz de Tkinter pero ocultarla (solo queremos el diálogo)
    root = tk.Tk()
    root.withdraw()
    # Abrir ventana en primer plano
    root.attributes('-topmost', True)

    # Abrir el explorador de archivos
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen de vehículo",
        filetypes=[
            ("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("Todos los archivos", "*.*")
        ]
    )

    # Cerrar la ventana raíz para liberar memoria
    root.destroy()

    return file_path


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    # 1. Verificar modelo
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        print("Por favor, ejecuta primero el script de entrenamiento.")
        input("Presiona Enter para salir...")
        return

    # 2. Cargar modelo (Solo se carga una vez al inicio)
    print("Cargando inteligencia artificial...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Modelo cargado exitosamente.\n")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return

    # 3. Bucle de predicción
    print("--- SISTEMA DE CLASIFICACIÓN OFFLINE ---")
    print("Selecciona una imagen de tu computadora.")
    print("Presiona 'Cancelar' en la ventana de selección para salir.\n")

    while True:
        # Abrir la ventana para elegir imagen
        ruta_imagen = seleccionar_archivo_gui()

        # Si el usuario presiona cancelar o cierra la ventana sin elegir nada
        if not ruta_imagen:
            print("Saliendo del programa...")
            break

        # Procesar la imagen seleccionada
        nombre, confianza = predecir_imagen(model, ruta_imagen)

        print("\n---------------- RESULTADO ----------------")
        print(f"Archivo: {os.path.basename(ruta_imagen)}")

        if nombre:
            print(f"El modelo detecta: {nombre.upper()}")
            print(f"Confianza: {confianza:.2f}%")
        else:
            print("No se pudo identificar la imagen. Intenta con otra.")

        print("------------------------------------------\n")
        # Pausa opcional si quieres ver el resultado antes de elegir la siguiente
        # input("Presiona Enter para analizar otra imagen...")


if __name__ == "__main__":
    main()