#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:27:09 2017

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy import fftpack

# se importa la imagen a estudiar
img = Image.open('leena512.bmp')
    
# se aplica la transformada de fourier a la imagen para extraer su espectro
# de frecuencias
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))  # centrado de frecuencias en torno a cero
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagen original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro de frecuencias'), plt.xticks([]), plt.yticks([])
plt.show()


# se guarda la imagen en un arreglo bidimensional
imagen = np.asarray(img,dtype=np.float32)


# componentes auxiliares 
matriz = []
nuevaImagen1 = []
nuevaImagen2 = []
listaAux1 = []
listaAux2 = []

# kernels que se aplicarán a las imagenes en el filtrado
kernelGauss = [[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]]
kernelBordes = [[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1],[1,2,0,-2,-1]]

# los arreglos de los kernels se transforman a arreglos de numpy
ListaGNumpyAux = np.array(kernelGauss)
ListaBNumpyAux = np.array(kernelBordes)

Gaussiano = ListaGNumpyAux
bordes = ListaBNumpyAux

# aplicacion del filtro gaussiano

for k in range(0,len(imagen)):
    listaAux1 = []
    for l in range(0,len(imagen[k])): 
        valor = 0
        for i in range(0,len(Gaussiano)):
            for j in range(0,len(Gaussiano)):         
                valor = valor + imagen[k-i][l-j]*Gaussiano[i][j]
        listaAux1.append(valor)     
    nuevaImagen1.append(listaAux1)        

# Normaliazcion de imagen filtrada 
        
imagenSuavizada = np.asarray(nuevaImagen1,dtype=np.float32)
maximo = imagenSuavizada.max()
imagenNormalizada1 = imagenSuavizada*(1/maximo)
imsave('FiltroSuavizado.png', imagenNormalizada1)

f = np.fft.fft2(imagenNormalizada1)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))  

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagen con filtrado Gausiano'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro de frecuencias'), plt.xticks([]), plt.yticks([])
plt.show()


# aplicacion del filtro bordes

for k in range(0,len(imagen)):
    listaAux2 = []
    for l in range(0,len(imagen[k])): 
        valor = 0
        for i in range(0,len(bordes)):
            for j in range(0,len(bordes)):         
                valor = valor + imagen[k-i][l-j]*bordes[i][j]
        listaAux2.append(valor)     
    nuevaImagen2.append(listaAux2)    

# Normaliazcion de imagen filtrada    
        
imagenBordes = np.asarray(nuevaImagen2,dtype=np.float32)
maximo = imagenBordes.max()
imagenNormalizada2 = imagenBordes*(1/maximo)
imsave('FiltroBordes.png', imagenNormalizada2)


f = np.fft.fft2(imagenNormalizada2)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))  

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Imagen con filtrado de bordes'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Espectro de frecuencias'), plt.xticks([]), plt.yticks([])
plt.show()

