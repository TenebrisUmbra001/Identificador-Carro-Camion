Contenido del README.md 

Copia el siguiente bloque y guárdalo como README.md en tu carpeta del proyecto: 
🚗 Clasificador de Vehículos IA (CIFAR-10)

Python 3.12TensorFlowLicense
👨‍🎓 Autor

Yohan Michel Perez Monzon
Ingeniería Informática - 3er Año
📝 Descripción del Proyecto

Este proyecto consiste en un sistema de Visión Artificial desarrollado en Python capaz de identificar y clasificar diferentes tipos de vehículos (automóviles y camiones) utilizando técnicas de Aprendizaje Supervisado (Deep Learning).

El sistema se entrena una vez utilizando datos en línea y luego funciona completamente Offline, permitiendo clasificar nuevas imágenes sin conexión a internet.
🧠 ¿Cómo funciona el código?

El núcleo del proyecto es una Red Neuronal Convolucional (CNN) construida con TensorFlow/Keras. El proceso se divide en dos fases:
1. Fase de Entrenamiento (entrenar_vehiculos.py)

    Adquisición de Datos: Se descarga el dataset público CIFAR-10, el cual contiene 60,000 imágenes de 32x32 píxeles etiquetadas en 10 categorías (Automóviles, Camiones, Aviones, etc.).
    Preprocesamiento: Las imágenes se normalizan (dividiendo los valores de píxeles por 255) para facilitar el cálculo matemático.
    Arquitectura del Modelo:
        Capas Conv2D: Detectan patrones visuales como bordes, ruedas y parabrisas.
        Capas MaxPooling: Reducen la dimensionalidad para retener las características más importantes y reducir el tiempo de cómputo.
        Capas Dense: Toman las características extraídas y deciden la clase final.
    Guardado: Una vez entrenado, la "inteligencia" (pesos de la red) se guarda en el archivo modelo_vehiculos.keras.

2. Fase de Predicción Offline (predecir_imagen.py)

    Carga del Modelo: El script lee el archivo .keras desde el disco duro, sin necesidad de conexión.
    Interfaz Gráfica: Utiliza tkinter para abrir una ventana nativa de selección de archivos.
    Procesamiento: Toma la imagen seleccionada por el usuario, la redimensiona a 32x32 píxeles y la normaliza.
    Inferencia: El modelo predice la clase y devuelve el resultado con un porcentaje de confianza.

🛠️ Instalación y Configuración

Este proyecto está optimizado para Python 3.12.

    Clonar o descargar el repositorio.
    Crear un entorno virtual (Recomendado):

    python -m venv .venv.venv\Scripts\activate

 

    Instalar las librerías necesarias:
    bash
     
      

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow numpy pillow matplotlib
     
     
      

🚀 Cómo usar el proyecto 
Paso 1: Entrenar (Se requiere Internet la primera vez) 

Ejecuta el script de entrenamiento para generar el modelo. 
bash
 
  
python entrenar_vehiculos.py
 
 
 

El resultado esperado es una precisión del ~70-72% y la creación del archivo modelo_vehiculos.keras. 

  
Paso 2: Predecir (Modo Offline) 

Una vez entrenado, puedes desconectar internet. Ejecuta el script de predicción. 
bash
 
  
python predecir_imagen.py
 
 
 

Se abrirá una ventana para que selecciones una imagen (JPG/PNG) de tu computadora. 


⚠️ Limitaciones del Modelo 

Es importante entender las restricciones de este prototipo escolar: 

    Resolución Baja: El modelo fue entrenado con imágenes de 32x32 píxeles. Si se suben fotos de muy alta resolución con muchos detalles ruidosos, la red puede perder precisión. 
    Ángulo de Cámara: El dataset original contiene principalmente imágenes en vista lateral. Las fotos frontales, aéreas o en perspectiva diagonal pueden generar clasificaciones erróneas. 
    Precisión: Con una precisión de ~72%, el modelo puede equivocarse 3 de cada 10 veces, especialmente si el fondo de la imagen es muy complejo. 
    Clases Limitadas: El modelo solo distingue 10 clases específicas del dataset CIFAR-10 (Avión, Auto, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión). 
    🔮 Futuras Mejoras 

Para expandir el proyecto en cursos superiores: 

     Implementar Data Augmentation para mejorar la precisión.
     Utilizar Transfer Learning con modelos pre-entrenados (ResNet, VGG16) para manejar imágenes de mayor resolución.
     Crear una interfaz gráfica completa (GUI) con PyQt en lugar de la consola.
     
