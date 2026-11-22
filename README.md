### Reconocedor de Números Escritos a Mano('CNN_Numeros.py')

Proyecto en Python que entrena una red neuronal con el dataset **MNIST** para reconocer digítos de "0" al "9" y luego utiliza el **OpenCV** para detectar objetos en tiempo real desde la cámara web y predecir que numeros se muestran.

### Requisitos Previos

*  **Python:** Se recomienda **Python 3.10 a 3.13.3**  
**Entorno virtual:** Recomendado para mantener aisladas las dependencias del proyecto.
* Camara web conectada o integrada


### Librerías necesarias 
* 'opencv-python'  ('cv2') | Captura video desde la cámara, escala de grises, umbralización, busqueda de contornos de rectángulos/texto y visualización.|
* 'numpy'| Manipulación de arreglos númericos, normalización y expansión de dimensiones.
* 'tensorflow'| Carga del dataset MNIST, construcción del modelo secuencial(`tf.keras`), entrenamiento, evalución y predicción. |

## Instalación y ejecución 
1.**Abre la terminal ** en la carpeta donde se encuentra 'CNN_Numeros.py'

2.**Crear el entorno virtual**
* Crear el entorno virtual llamado venv:
```powershell 
python -m venv venv
```

3. **Activa tu entorno virtual**
    * **Windows (CMD):**
	```cmd 
	venv\Scripts\activate
	```
	* **Windows (PowerShell):**
	```powershell 
	.\venv\Scripts\Activate.ps1
	``` 
	* **mac0s / Linux: **
	```bash 
	source venv/bin/activate
	``` 

3.**Instala las dependencias (si aún no lo hiciste):**
* Una vez creado el entorno virtual activado, instalar opencv-python, numpy y tensorflow.
```powershell 
pip install numpy
```
* Instalar OpenCV para el manejo de la cámara y el procesamiento de imagen:
```
pip install opencv-python 
``` 
```
pip install tensorflow 
```
* Ejecución del Scipt:
```
python CNN_Numeros.py
```

Lo que el programa hace es abrir tu camara integrada (si es que el dispositivo cuenta con una) con "cv2.VideoCaptura(0)",  despues imprime en la consola las instrucciones:
- "Inicia las 5 epocas o entrenamientas que le pusimos al programa.
-  1.Escribe en una hoja un numero .
-  2.Pon la hoja frente a la camara.
Abre una ventana donde se muestra el numero que esta reconociendo | La Confianza es decir lo preciso que es.
- PRESIONA Q PARA SALIR O TERMINAR EL PROGRAMA.
