DIFERENCIACION DE CARROS Y CAMIONES EN IMÁGENES CON APRENDIZAJE SUPERVISADO

Proyecto: Agente de Clasificación de Vehículos 
Autor: [Yohan Michel Perez Monzon]
Carrera: Ingeniería Informática
Año: 3er Año 
                                                                                                                      Introducción 




El objetivo principal es desarrollar un "agente" inteligente que, tras un proceso de aprendizaje supervisado, sea capaz de recibir una imagen proporcionada por un usuario, analizar sus patrones visuales y predecir a qué categoría pertenece con un grado de confianza determinado. Para ello, se utiliza el dataset estándar CIFAR-10 y la biblioteca TensorFlow/Keras para implementar una Red Neuronal Convolucional (CNN). 
                                                                                                                     Desarrollo 

La solución se divide en dos fases distintas: entrenamiento e inferencia (uso del agente).  
Fase de Entrenamiento: Se utiliza una red neuronal convolucional (CNN). Las CNN son el estándar en visión por computadora porque imitan la corteza visual humana, detectando características locales como bordes, texturas y formas, y luego combinándolas para identificar objetos complejos. 
Fase de Inferencia: Una vez entrenado el modelo, este se guarda en disco. El segundo script actúa como el "agente", cargando el conocimiento adquirido y aplicándolo a nuevas imágenes sin conexión a internet. 

 Implementación del Entrenamiento 
Para el entrenamiento, se utilizó el dataset CIFAR-10, el cual contiene 60,000 imágenes a color de 32x32 píxeles divididas en 10 clases. 

    
   Aqui el codigo de la construccion del modelo 
 def construir_modelo():
    model = models.Sequential([
    
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),  
        layers.Dense(64, activation='relu'), 
        layers.Dense(10)  
    ])
    return model
    
Explicación del funcionamiento: 

     Conv2D: Aplica filtros a la imagen para extraer características (bordes, curvas).
     MaxPooling2D: Reduce el tamaño de la imagen calculando el máximo valor en una ventana, lo que reduce el costo computacional y ayuda a evitar el sobreajuste.
     Dense (Capa densa): Las últimas capas toman las características extraídas y deciden la clasificación final.
     

El segundo componente del software es el agente que interactúa con el usuario. Este script no requiere internet, ya que carga el archivo modelo_vehiculos.keras generado previamente. 

El flujo de trabajo del agente es el siguiente: 

    Carga del Modelo: Se lee el archivo binario que contiene los pesos de la red neuronal. 
    Interfaz de Usuario (GUI): Se utiliza la librería tkinter para abrir un explorador de archivos y permitir la selección de una imagen local. 
    Pre-procesamiento: La imagen seleccionada por el usuario (que puede tener cualquier tamaño) debe transformarse para coincidir con el formato de entrada del modelo (32x32 píxeles y valores normalizados entre 0 y 1). 
 
def predecir_imagen(model, image_path):
   Aqui el codigo para que pueda predecir la imagen enviada por el usuario
    img = Image.open(image_path).convert('RGB').resize((32, 32))
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

   
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    return class_names[np.argmax(score)], 100 * np.max(score)
 
 
  Ejemplos de Uso 

El sistema permite dos flujos de uso principales: 

    Entrenamiento: 
         El ingeniero ejecuta el primer script.
         El sistema descarga CIFAR-10 y muestra el progreso de precisión (accuracy) epoch tras epoch.
         Al finalizar, se genera el archivo modelo_vehiculos.keras.
          

    Operación del Agente: 
         El usuario ejecuta el segundo script.
         Aparece una ventana emergente para buscar una imagen (ej. una foto de un auto guardada en el escritorio).
         
        El agente responde en la consola:
        ---------------- RESULTADO ----------------
        Archivo: mi_auto.jpg
        El modelo detecta: AUTOMÓVIL
        Confianza: 85.42% 
        text
         
          

                                                                                                                         
Conclusiones                      
El desarrollo de este proyecto ha permitido consolidar conocimientos prácticos sobre el ciclo de vida de un sistema de Inteligencia Artificial, desde la adquisición de datos hasta el despliegue de un modelo funcional. 

     Eficacia de las CNN: Se comprobó que las Redes Neuronales Convolucionales son herramientas potentes para la clasificación de imágenes, logrando aprender patrones visuales complejos con una arquitectura relativamente sencilla (aprox. 3 capas convolucionales).
     Portabilidad y Offline: Al separar la fase de entrenamiento de la fase de inferencia y guardar el modelo en formato .keras, se logró crear un agente autónomo. Esto es crítico en aplicaciones reales donde el acceso a internet puede ser limitado o donde la privacidad de los datos requiere que el procesamiento sea local.
     Limitaciones y Futuro: El modelo actual trabaja con resoluciones bajas (32x32). Para un entorno de producción real o ingeniería avanzada, se recomienda el uso de Transfer Learning (usar modelos pre-entrenados como ResNet o VGG16) para mejorar la precisión en imágenes de alta resolución.
     
