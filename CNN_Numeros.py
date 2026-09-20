##############################
#Reconocedor de numeros a mano
##############################

####################################
#Importar OpenCV para usar la camara
import cv2

#Libreria para trabajar matrices
import numpy as np

#Importar TensorFlow para crear la red neuronal
import tensorflow as tf

###############################
#Cargar la base de datos

# MNIST que contiene imagenes muestra de numeros escritos a mano de 0 a 9
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Dataset MNIST cargado")

# Las imagenes tienen dimension de 28x28 pixeles
# Los valores de cada pixel van de 0 a 255
# Los divido cada pixel entre 255 (rango de pixeles de 0 a 1)

x_train = x_train / 255.0
x_test = x_test / 255.0

# Las etiquetas son numeros del 0 al 9, no imagenes
# por eso no se normalizan con 255

########################
# Creacion de la red neuronal 

#Sequencial permite crear una red neuronal colocando las capas tras capa
modelo = tf.keras.Sequential([
    #Convertir la imagen de 28x28 a una sola lista
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    #Crear primera capa neuronal
    #128 neuronas (ReLu)
    tf.keras.layers.Dense(128, activation='relu'),

    # Segunda Capa neuronal
    #64 neuronas 
    tf.keras.layers.Dense(64, activation='relu'),

    # Capa de salida 
    #10 posibilidades (0,1,2,3,4,5,6,7,8,9)
    tf.keras.layers.Dense(10, activation='softmax')

])

##############################
#Preparar el entrenamiento

#Configuracion de como aprendera la red 
modelo.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


###########################3
#Entrenar la red neuronal

print("Entrenando la red neuronal....")
# epocas indica cuantas veces recorrera el conjunto del entrenamiento

modelo.fit(
    x_train, 
    y_train,
    epochs = 5
)

print("Entrenamiento Terminado.........")

###################################
# Evaluar la red neuronal 

#Comprobar que tan bien funciona utilizando imagenes que no vio durante el entrenamiento

resultado = modelo.evaluate(x_test, y_test)

print("Evaluacion terminada")
print("Precision: ", resultado[1])

#################################
#Iniciar Cámara 
camara = cv2.VideoCapture(0) #El cero es la camara integrada 

print("Camara iniciada")
print(" 1.Escribe en una hoja un numero ")
print("2. Pon la hoja frente a la camara")
print("PRESIONA Q PARA SALIR")

##################################
#Programa Principal
while True:

    #Captura del fram con la camara
    ret, frame = camara.read()

    #Sino se puede obtener la imagen se termina el programa
    if not ret:
        print("No se pudo acceder a la camara")
        break

    #Convertir a escala de grises 
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Crear el umbral de deteccion del numero

    _, binaria = cv2.threshold(gris, 100, 255, cv2.THRESH_BINARY_INV)

    #Buscar los objetos de imagen 
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Revisar cada uno de los contornos
    for contorno in contornos:
        x,y, ancho, alto = cv2.boundingRect(contorno)
        #Ignorar objetos pequeños
        if ancho < 20 or alto < 20:
            continue 
        #Dibujar rectangulo
        cv2.rectangle(frame, (x,y), (x+ancho, y+alto), (0,255,0),)

        #Extraer el numero 
        numero = binaria[y:y+alto, x:x+ancho]

        #Redimensionar a 28x28 para usar MNIST
        numero = cv2.resize(numero, (28, 28))

        #Convertimos de 0 a 1 los valores de la matriz
        numero = numero / 255.0

        #Agregamos una dimension porque MNIST acepta solo listas
        numero = np.expand_dims(numero, axis=0)

        #Predecir el valor con CNN
        prediccion = modelo.predict(numero, verbose = 0)

        #Obtener el valor con mayor probabilidad
        resultado = np.argmax(prediccion)

        #Obtener la confianzaa 
        confianza = np.max(prediccion)*100

        #Mostrar el resultado en la pantalla
        texto = f"{resultado}, ({confianza:.1f}%)"

        cv2.putText(frame, texto, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        #Mostar el resultaod en terminal
        print(f"Numero reconocido: {resultado} |" f"Confianza: {confianza:.1f}% ")

        #Mostrar la camara
        cv2.imshow("Reconocedor de numeros", frame)

        #Salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Cerrar todo
camara.release()
cv2.destroyAllWindows()
print("Programa terminado")