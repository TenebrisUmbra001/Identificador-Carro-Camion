import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import numpy as np
import os


# ==========================================
# SECCIÓN 1: CONFIGURACIÓN Y DESCARGA DE DATOS
# ==========================================
def descargar_y_preparar_datos():
    print("Descargando dataset CIFAR-10 (esto requiere internet)...")
    # CIFAR-10 contiene 60,000 imágenes de 32x32 en 10 clases.
    # Clase 1: Automóvil, Clase 9: Camión (Objetivos principales)
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # Normalizamos los valores de píxeles (0-255) para que estén entre 0 y 1
    # Esto ayuda a que la red neuronal aprenda más rápido.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Nombres de las clases para referencia futura
    class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
                   'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

    return x_train, y_train, x_test, y_test, class_names


# ==========================================
# SECCIÓN 2: CONSTRUCCIÓN DEL MODELO (CNN)
# ==========================================
def construir_modelo():
    model = models.Sequential([
        # Capa Convolucional: Detecta bordes y formas
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),  # Reduce el tamaño manteniendo lo importante

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.Flatten(),  # Convierte la imagen 2D en un vector 1D
        layers.Dense(64, activation='relu'),  # Capa totalmente conectada
        layers.Dense(10)  # Capa de salida (10 clases posibles)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


# ==========================================
# SECCIÓN 3: ENTRENAMIENTO Y EVALUACIÓN
# ==========================================
def entrenar_y_evaluar(model, x_train, y_train, x_test, y_test):
    print("Iniciando entrenamiento... (esto puede tardar unos minutos)")
    # Entrenamos por 10 épocas (pasadas completas sobre los datos)
    history = model.fit(x_train, y_train, epochs=10,
                        validation_data=(x_test, y_test))

    print("Evaluando modelo final...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nPrecisión del modelo en datos de prueba: {test_acc:.2%}')
    return history


# ==========================================
# SECCIÓN 4: GUARDADO DEL MODELO
# ==========================================
def guardar_modelo(model, filename='modelo_vehiculos.keras'):
    print(f"Guardando modelo entrenado en {filename}...")
    model.save(filename)
    print("Modelo guardado con éxito. Ahora puedes desconectarte de internet.")


# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Obtener datos
    x_train, y_train, x_test, y_test, class_names = descargar_y_preparar_datos()

    # 2. Crear modelo
    model = construir_modelo()
    model.summary()

    # 3. Entrenar
    entrenar_y_evaluar(model, x_train, y_train, x_test, y_test)

    # 4. Guardar
    guardar_modelo(model)