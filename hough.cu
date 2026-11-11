/*
 ============================================================================
 Archivo       : hough.cu
 Autor base    : G. Barlas (version 1.0, diciembre 2014)
 Modificado por: Mario Rocha - Proyecto 3, Computacion Paralela, UVG
 Licencia      : GNU GPL 3.0
 Descripcion   : Transformada lineal de Hough en CUDA. Cada pixel encendido
                 de una imagen PGM vota por todas las rectas (r, theta) que
                 pasan por el. Se calcula en CPU (referencia) y en GPU, se
                 comparan los acumuladores y se mide el tiempo del kernel
                 con CUDA events. Hay un kernel por tipo de memoria:
                   global     - tablas cos/sin y acumulador en memoria Global
                   constante  - tablas cos/sin en memoria Constante
                   compartida - Constante + acumulador local por bloque en
                                memoria Compartida
 Compilar      : make
 Ejecutar      : ./hough <imagen.pgm> [global|constante|compartida]
                 (default: global)
 ============================================================================

 Estilo de documentacion
 -----------------------
 - Cada archivo inicia con este encabezado (archivo, autor, descripcion).
 - Cada funcion lleva un bloque con:
       Funcion / Descripcion / Parametros / Retorno
 - Dentro de las funciones, cada etapa inicia con "// --- Etapa ---" y los
   detalles se explican con comentarios de linea "//".
 - Los pendientes se marcan como "TODO (<tarea>):".
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

// Discretizacion del espacio de Hough
const int degreeInc = 2;                       // incremento de theta en grados
const int degreeBins = 180 / degreeInc;        // 90 valores de theta en [0, 180)
const int rBins = 100;                         // 100 valores de r en [-rMax, rMax]
const float radInc = degreeInc * M_PI / 180;   // incremento de theta en radianes

/*****************************************************************
 * Funcion    : CPU_HoughTran
 * Descripcion: Version secuencial de la transformada. Sirve como
 *              referencia para validar el resultado de la GPU.
 * Parametros : pic - pixeles de la imagen (w * h, fila por fila)
 *              w   - ancho de la imagen
 *              h   - alto de la imagen
 *              acc - (salida) puntero al acumulador que se reserva
 *                    aqui con new[]; el llamador lo libera con delete[]
 * Retorno    : ninguno
 *****************************************************************/
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  // --- Parametros del espacio (r, theta) ---
  // rMax: distancia del centro a una esquina = sqrt(w^2 + h^2) / 2
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  // acumulador de rBins * degreeBins = 100 * 90 = 9000 celdas, en ceros
  *acc = new int[rBins * degreeBins];
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  // ancho de cada bin de r: el rango [-rMax, rMax] se divide en rBins partes
  float rScale = 2 * rMax / rBins;

  // --- Votacion: cada pixel encendido vota por todas sus rectas ---
  for (int i = 0; i < w; i++)        // i = columna
    for (int j = 0; j < h; j++)      // j = fila
      {
        int idx = j * w + i;         // indice lineal del pixel (i, j)
        if (pic[idx] > 0)
          {
            // origen en el centro de la imagen; Y se invierte porque las
            // filas crecen hacia abajo
            int xCoord = i - xCent;
            int yCoord = yCent - j;
            float theta = 0;
            for (int tIdx = 0; tIdx < degreeBins; tIdx++)
              {
                // r = x cos(theta) + y sin(theta), ecuacion (1.3)
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                // mapea r de [-rMax, rMax] al indice [0, rBins)
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++;   // un voto para (r, theta)
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// Tablas de senos y cosenos en memoria Constante.
// Se declaran aqui, fuera de main, para que tengan scope global y los
// kernels las lean sin recibirlas como parametro. El host las llena con
// cudaMemcpyToSymbol; no se reservan con cudaMalloc ni se liberan con
// cudaFree. Se llaman c_Cos/c_Sin (y no d_Cos/d_Sin como en el
// enunciado) porque d_Cos/d_Sin ya son los punteros en memoria Global
// que usa el kernel GPU_HoughTran, y los parametros con ese nombre
// taparian a estas tablas.
// Tamano: 2 * 90 floats = 720 bytes, muy por debajo del limite de 64 KB.
__constant__ float c_Cos[degreeBins];
__constant__ float c_Sin[degreeBins];

/*****************************************************************
 * Funcion    : GPU_HoughTran  (kernel, memoria Global)
 * Descripcion: Un hilo por pixel. Si el pixel esta encendido, vota
 *              en el acumulador global por las degreeBins rectas que
 *              pasan por el.
 * Parametros : pic    - imagen en memoria global del device
 *              w, h   - ancho y alto de la imagen
 *              acc    - acumulador en memoria global (inicializado en 0
 *                       por el host)
 *              rMax   - radio maximo (centro -> esquina)
 *              rScale - ancho de cada bin de r
 *              d_Cos  - tabla de cos(theta) precalculada, en memoria global
 *              d_Sin  - tabla de sin(theta) precalculada, en memoria global
 * Retorno    : ninguno (el resultado queda en acc)
 *****************************************************************/
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // --- ID global del hilo ---
  // Grid 1D de bloques 1D: los bloques anteriores ya cubrieron
  // blockIdx.x * blockDim.x pixeles, y threadIdx.x es la posicion
  // dentro del bloque.
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  // el ultimo bloque puede tener hilos de sobra (indices validos: 0 .. w*h-1)
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;

  // --- Indice lineal -> coordenadas cartesianas ---
  // La imagen esta guardada fila por fila, asi que:
  //   columna = gloID % w      fila = gloID / w
  // Luego se mueve el origen al centro de la imagen y se invierte Y,
  // porque en la imagen las filas crecen hacia abajo y en el plano
  // cartesiano Y crece hacia arriba:
  //
  //   (0,0)----------> columna                 y
  //     |                                      ^
  //     |       + (xCent, yCent)     ==>       |
  //     |                                 -----+-----> x
  //     v fila                                 |
  //
  // Con el origen en el centro, r queda en el rango [-rMax, rMax].
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // --- Votacion ---
  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // version alternativa para comparar tiempos (cos/sin en cada hilo):
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx);
          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          // Hay un hilo por pixel, pero todos votan en el mismo acumulador:
          // los pixeles de una misma recta caen en la misma celda
          // (rIdx, tIdx). acc[k]++ es leer-sumar-escribir; si dos hilos leen
          // el mismo valor a la vez, uno de los votos se pierde. atomicAdd
          // hace la suma como una sola operacion indivisible.
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

/*****************************************************************
 * Funcion    : GPU_HoughTranConst  (kernel, memoria Global + Constante)
 * Descripcion: Igual a GPU_HoughTran, pero lee cos(theta) y sin(theta)
 *              de las tablas c_Cos/c_Sin en memoria Constante en lugar
 *              de recibirlas como punteros a memoria Global.
 * Parametros : pic    - imagen en memoria global del device
 *              w, h   - ancho y alto de la imagen
 *              acc    - acumulador en memoria global (inicializado en 0
 *                       por el host)
 *              rMax   - radio maximo (centro -> esquina)
 *              rScale - ancho de cada bin de r
 * Retorno    : ninguno (el resultado queda en acc)
 *****************************************************************/
__global__ void GPU_HoughTranConst (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  // --- ID global del hilo (mismo calculo que en GPU_HoughTran) ---
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;

  // --- Indice lineal -> coordenadas cartesianas (origen al centro) ---
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // --- Votacion ---
  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // En cada iteracion todos los hilos del warp leen la misma
          // posicion c_Cos[tIdx] / c_Sin[tIdx]. La memoria Constante hace
          // broadcast de ese valor a todo el warp en una sola lectura.
          float r = xCoord * c_Cos[tIdx] + yCoord * c_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          // varios pixeles votan por la misma celda: suma atomica
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }
}

/*****************************************************************
 * Funcion    : GPU_HoughTranShared  (kernel, Global + Constante +
 *              Compartida)
 * Descripcion: Cada bloque vota primero en un acumulador local
 *              localAcc en memoria Compartida y, al final, suma ese
 *              acumulador local al acumulador global acc. Asi la
 *              mayoria de los atomicAdd ocurren en memoria Compartida
 *              (on-chip) y no en memoria Global. Las tablas cos/sin se
 *              leen de memoria Constante (c_Cos/c_Sin).
 * Parametros : pic    - imagen en memoria global del device
 *              w, h   - ancho y alto de la imagen
 *              acc    - acumulador en memoria global (inicializado en 0
 *                       por el host)
 *              rMax   - radio maximo (centro -> esquina)
 *              rScale - ancho de cada bin de r
 * Retorno    : ninguno (el resultado queda en acc)
 *****************************************************************/
__global__ void GPU_HoughTranShared (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  // --- IDs del hilo ---
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;   // pixel que procesa
  int locID = threadIdx.x;                             // posicion en el bloque

  // --- Acumulador local del bloque en memoria Compartida ---
  // degreeBins * rBins = 9000 ints = 36 KB, cabe en los 48 KB de memoria
  // compartida por bloque. Solo lo ven los hilos de este bloque.
  __shared__ int localAcc[degreeBins * rBins];

  // --- Inicializacion de localAcc en 0 ---
  // La memoria compartida no se puede inicializar desde el host. Los 256
  // hilos del bloque se reparten las 9000 celdas: el hilo locID limpia
  // locID, locID + blockDim.x, locID + 2 * blockDim.x, ...
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
    localAcc[i] = 0;

  // Barrera 1: nadie vota hasta que todo localAcc este en 0.
  __syncthreads ();

  // --- Votacion en el acumulador local ---
  // Aqui NO se usa "return" para los hilos sobrantes del ultimo bloque
  // (como en los otros kernels): todos los hilos del bloque deben llegar
  // a las dos barreras, o el comportamiento de __syncthreads() es
  // indefinido. Por eso se usa la condicion gloID < w * h.
  if (gloID < w * h && pic[gloID] > 0)
    {
      int xCent = w / 2;
      int yCent = h / 2;
      // indice lineal -> coordenadas cartesianas (origen al centro)
      int xCoord = gloID % w - xCent;
      int yCoord = yCent - gloID / w;

      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          float r = xCoord * c_Cos[tIdx] + yCoord * c_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          // varios hilos del bloque votan por la misma celda: suma
          // atomica, pero sobre memoria Compartida (mucho mas rapida)
          atomicAdd (localAcc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  // Barrera 2: nadie copia al global hasta que todos hayan votado.
  __syncthreads ();

  // --- Suma del acumulador local al global ---
  // Mismo reparto que en la inicializacion. Varios bloques suman a la
  // misma celda de acc, por eso tambien es atomicAdd. Las celdas en 0 se
  // saltan porque no aportan nada y evitan accesos a memoria Global.
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
    if (localAcc[i] > 0)
      atomicAdd (acc + i, localAcc[i]);
}

/*****************************************************************
 * Funcion    : main
 * Descripcion: Lee la imagen, calcula la transformada en CPU y en GPU
 *              con el kernel elegido, mide el tiempo del kernel, compara
 *              ambos acumuladores y libera toda la memoria.
 * Parametros : argc, argv - argv[1] es la ruta de la imagen .pgm
 *                           argv[2] (opcional) es el kernel a usar:
 *                           global (default), constante o compartida
 * Retorno    : 0 si todo salio bien, 1 si hubo un error
 *****************************************************************/
int main (int argc, char **argv)
{
  int i;

  // --- Argumentos: imagen y kernel ---
  if (argc < 2)
  {
    fprintf (stderr, "Uso: %s <imagen.pgm> [global|constante|compartida]\n", argv[0]);
    return 1;
  }

  // Se valida antes de reservar memoria para no dejar nada sin liberar.
  const char *version = (argc >= 3) ? argv[2] : "global";
  if (strcmp (version, "global") != 0 && strcmp (version, "constante") != 0
      && strcmp (version, "compartida") != 0)
  {
    fprintf (stderr, "Kernel desconocido: %s (usar global, constante o compartida)\n", version);
    return 1;
  }
  printf ("Kernel: %s\n", version);

  // --- Lectura de la imagen ---
  PGMImage inImg (argv[1]);   // el destructor libera inImg.pixels

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // --- Version CPU (referencia) ---
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // --- Tablas de senos y cosenos ---
  // Se calculan una sola vez en el host para no evaluar funciones
  // trigonometricas en cada hilo.
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Memoria Global (kernel global): copia normal a los punteros de cudaMalloc.
  cudaMemcpy (d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy (d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

  // Memoria Constante (kernels constante y compartida): las tablas son simbolos
  // __constant__, asi que se copian con cudaMemcpyToSymbol.
  cudaMemcpyToSymbol (c_Cos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol (c_Sin, pcSin, sizeof (float) * degreeBins);

  // --- Reserva y copia de datos host -> device ---
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // --- Configuracion del grid ---
  // Grid 1D de bloques de 256 hilos, un hilo por pixel.
  // (w*h + 255) / 256 redondea hacia arriba con aritmetica entera.
  int blockNum = (w * h + 255) / 256;

  // --- Lanzamiento del kernel y medicion con CUDA events ---
  // Los eventos se registran en el stream de la GPU, asi que se mide
  // solo el tiempo del kernel elegido y no el del host.
  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);
  if (strcmp (version, "constante") == 0)
    GPU_HoughTranConst <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  else if (strcmp (version, "compartida") == 0)
    GPU_HoughTranShared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  else
    GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord (stop);
  cudaEventSynchronize (stop);   // esperar a que el kernel termine

  cudaError_t err = cudaGetLastError ();
  if (err != cudaSuccess)
  {
    fprintf (stderr, "Error en el kernel: %s\n", cudaGetErrorString (err));
    return 1;
  }

  float elapsedMs = 0;
  cudaEventElapsedTime (&elapsedMs, start, stop);
  printf ("Tiempo kernel: %f ms\n", elapsedMs);

  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  // --- Resultados device -> host ---
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // --- Validacion contra la version CPU ---
  // Pueden aparecer pocas diferencias de +-1 entre celdas vecinas de r:
  // la GPU usa FMA y la CPU calcula cos/sin en double, y un r que cae
  // justo en el borde de un bin puede quedar en el bin de al lado.
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");

  // --- Liberacion de memoria y reinicio del dispositivo ---
  cudaFree (d_in);       // device: cudaMalloc -> cudaFree
  cudaFree (d_hough);
  cudaFree (d_Cos);
  cudaFree (d_Sin);
  // c_Cos/c_Sin son memoria Constante estatica: no se liberan con
  // cudaFree; se liberan con el contexto en cudaDeviceReset().
  free (pcCos);          // host: malloc -> free
  free (pcSin);
  free (h_hough);
  delete[] cpuht;        // host: new[] -> delete[]
  cudaDeviceReset ();    // libera el contexto y los recursos de la GPU

  return 0;
}
