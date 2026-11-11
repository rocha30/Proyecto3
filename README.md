# Proyecto 3: Transformada lineal de Hough en CUDA

Computación Paralela y Distribuida, UVG. Mario Rocha y Luis Pedro Lira.

El programa detecta líneas rectas en una imagen de bordes (`runway.pgm`) con la transformada lineal de Hough. Hay tres versiones del kernel, una por cada combinación de memorias de la GPU:

| Versión | Kernel | Memorias |
|---|---|---|
| `global` | `GPU_HoughTran` | Global (imagen, tablas cos/sin y acumulador) |
| `constante` | `GPU_HoughTranConst` | Global + Constante (tablas cos/sin) |
| `compartida` | `GPU_HoughTranShared` | Global + Constante + Compartida (acumulador local por bloque) |

## Compilar

Se necesita `nvcc` (CUDA Toolkit) y `g++`. Se probó en Google Colab con una GPU Tesla T4.

```
make
```

## Ejecutar

```
./hough runway.pgm global       # default
./hough runway.pgm constante
./hough runway.pgm compartida
```

El programa:
1. Binariza la imagen: un pixel está encendido si su gris es mayor que 25 (`pixelThreshold`). `runway.pgm` es una imagen de bordes en escala de grises y la mayoría de sus pixeles con gris > 0 son ruido muy tenue; sin este paso las rectas ganadoras son las que cruzan la zona con ruido y no los bordes reales.
2. Calcula la transformada en CPU (referencia) y en GPU con el kernel elegido.
3. Imprime `Tiempo kernel: <ms> ms`, medido con CUDA events.
4. Compara los acumuladores de CPU y GPU. Pueden aparecer unas pocas diferencias de ±1 voto entre celdas vecinas de r, por el redondeo de las instrucciones FMA de la GPU; son iguales en las tres versiones.
5. Genera `output/lineas.png`: la imagen de entrada con las líneas detectadas en rojo. Se dibujan las celdas del acumulador con más votos que *promedio + 2 · desviación estándar* que además son máximos locales, hasta un máximo de 10 (`maxLines`), las de más votos.

## Bitácora de tiempos

```
chmod +x run_bitacora.sh
CUDA_MODULE_LOADING=EAGER ./run_bitacora.sh global
CUDA_MODULE_LOADING=EAGER ./run_bitacora.sh constante
CUDA_MODULE_LOADING=EAGER ./run_bitacora.sh compartida
```

Cada comando corre el programa 10 veces, guarda `bitacora/<version>.csv` (`corrida,tiempo_ms`) e imprime el promedio y la desviación estándar.

## Archivos

| Archivo | Contenido |
|---|---|
| `hough.cu` | Versión CPU, los tres kernels y `main` |
| `common/pgm.h`, `common/pgm.cpp` | Lectura de imágenes PGM |
| `common/lineas.h`, `common/lineas.cpp` | Dibujo de las líneas detectadas y escritura del PNG |
| `common/stb_image_write.h` | Librería de dominio público para escribir PNG ([stb](https://github.com/nothings/stb)) |
| `run_bitacora.sh` | Script de mediciones |
| `bitacora/` | Tiempos de las 10 corridas de cada kernel |
