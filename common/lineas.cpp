/*
 ============================================================================
 Archivo     : lineas.cpp
 Autor       : Luis Pedro Lira - Proyecto 3, Computacion Paralela, UVG
 Modificado  : Mario Rocha - seleccion de las maxLineas con mas votos
 Descripcion : Implementacion de DrawLines (ver lineas.h). Usa la libreria
               de un solo header stb_image_write (dominio publico, S. Barrett)
               para escribir el PNG.
 ============================================================================
 */
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "lineas.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include <sys/stat.h>

/*****************************************************************
 * Funcion    : EsMaximoLocal
 * Descripcion: Indica si la celda (rIdx, tIdx) tiene al menos tantos
 *              votos como sus 8 vecinas en el acumulador.
 *              theta es circular: el vecino de tIdx = degreeBins - 1 es
 *              tIdx = 0, pero con r con signo contrario (la recta con
 *              theta + 180 y r es la misma que con theta y -r). Por eso,
 *              al dar la vuelta en theta, el bin de r se refleja:
 *              rIdx -> rBins - 1 - rIdx.
 * Parametros : acc        - acumulador (rIdx * degreeBins + tIdx)
 *              rIdx, tIdx - celda a revisar
 *              degreeBins - numero de bins de theta
 *              rBins      - numero de bins de r
 * Retorno    : true si es maximo local, false si alguna vecina es mayor
 *****************************************************************/
static bool EsMaximoLocal (int *acc, int rIdx, int tIdx, int degreeBins, int rBins)
{
  int votos = acc[rIdx * degreeBins + tIdx];

  for (int dr = -1; dr <= 1; dr++)
    for (int dt = -1; dt <= 1; dt++)
      {
        if (dr == 0 && dt == 0)
          continue;

        int r = rIdx + dr;
        int t = tIdx + dt;

        // vuelta en theta: se refleja el bin de r
        if (t < 0 || t >= degreeBins)
          {
            t = (t + degreeBins) % degreeBins;
            r = rBins - 1 - r;
          }

        if (r < 0 || r >= rBins)
          continue;

        if (acc[r * degreeBins + t] > votos)
          return false;
      }

  return true;
}

/*****************************************************************
 * Funcion    : PintarPixel
 * Descripcion: Pinta de rojo el pixel (i, j) del buffer RGB si cae
 *              dentro de la imagen.
 * Parametros : rgb  - buffer de w * h * 3 bytes
 *              w, h - ancho y alto de la imagen
 *              i, j - columna y fila del pixel
 * Retorno    : ninguno
 *****************************************************************/
static void PintarPixel (unsigned char *rgb, int w, int h, int i, int j)
{
  if (i < 0 || i >= w || j < 0 || j >= h)
    return;

  unsigned char *p = rgb + 3 * (j * w + i);
  p[0] = 255;   // R
  p[1] = 0;     // G
  p[2] = 0;     // B
}

/*****************************************************************
 * Funcion    : DrawLines
 * Descripcion: Ver lineas.h.
 *****************************************************************/
int DrawLines (const char *fname, unsigned char *pic, int w, int h,
               int *acc, int degreeBins, int rBins, float radInc,
               float rMax, float rScale, int maxLineas)
{
  int nCeldas = degreeBins * rBins;

  // --- Threshold: promedio + 2 desviaciones estandar ---
  // Se usa la desviacion estandar poblacional porque se toman todas las
  // celdas del acumulador, no una muestra.
  double suma = 0, sumaCuad = 0;
  for (int k = 0; k < nCeldas; k++)
    {
      suma += acc[k];
      sumaCuad += (double) acc[k] * acc[k];
    }
  double promedio = suma / nCeldas;
  double desv = sqrt (sumaCuad / nCeldas - promedio * promedio);
  double threshold = promedio + 2 * desv;

  // --- Copia de la imagen en gris a RGB (R = G = B) ---
  unsigned char *rgb = (unsigned char *) malloc ((size_t) w * h * 3);
  if (rgb == NULL)
    {
      fprintf (stderr, "DrawLines: no se pudo reservar el buffer RGB\n");
      return -1;
    }
  for (int k = 0; k < w * h; k++)
    {
      rgb[3 * k]     = pic[k];
      rgb[3 * k + 1] = pic[k];
      rgb[3 * k + 2] = pic[k];
    }

  // --- Candidatas: sobre el threshold y maximos locales ---
  int *cand = (int *) malloc (sizeof (int) * nCeldas);
  if (cand == NULL)
    {
      fprintf (stderr, "DrawLines: no se pudo reservar la lista de candidatas\n");
      free (rgb);
      return -1;
    }
  int nCand = 0;
  for (int k = 0; k < nCeldas; k++)
    if (acc[k] > threshold && EsMaximoLocal (acc, k / degreeBins, k % degreeBins, degreeBins, rBins))
      cand[nCand++] = k;

  // --- Seleccion de las maxLineas candidatas con mas votos ---
  // Seleccion parcial: en cada paso se mueve al frente la de mas votos
  // entre las que quedan. Son pocas candidatas, asi que basta con esto.
  int nSel = (nCand < maxLineas) ? nCand : maxLineas;
  for (int a = 0; a < nSel; a++)
    {
      int mejor = a;
      for (int b = a + 1; b < nCand; b++)
        if (acc[cand[b]] > acc[cand[mejor]])
          mejor = b;
      int tmp = cand[a]; cand[a] = cand[mejor]; cand[mejor] = tmp;
    }

  // --- Dibujo de las rectas seleccionadas ---
  int xCent = w / 2;
  int yCent = h / 2;
  int nLineas = 0;

  for (int q = 0; q < nSel; q++)
    {
      int rIdx = cand[q] / degreeBins;
      int tIdx = cand[q] % degreeBins;

      // (rIdx, tIdx) -> (r, theta). r es el centro del bin: el kernel
      // calcula rIdx = (r + rMax) / rScale, asi que el bin rIdx cubre
      // [rIdx * rScale - rMax, (rIdx + 1) * rScale - rMax).
      float theta = tIdx * radInc;
      float r = rIdx * rScale - rMax + rScale / 2;
      float c = cos (theta);
      float s = sin (theta);

      // La recta es x cos(theta) + y sin(theta) = r en coordenadas
      // centradas. Para pasar a pixel se usa la inversa de xCoord/yCoord
      // del kernel: i = x + xCent, j = yCent - y.
      // Se recorre el eje en el que la recta avanza mas rapido, para que
      // quede continua (sin huecos):
      if (fabs (s) > fabs (c))
        {
          // recta mas horizontal: una fila j por cada columna i
          for (int i = 0; i < w; i++)
            {
              float x = i - xCent;
              float y = (r - x * c) / s;
              PintarPixel (rgb, w, h, i, (int) lroundf (yCent - y));
            }
        }
      else
        {
          // recta mas vertical: una columna i por cada fila j
          for (int j = 0; j < h; j++)
            {
              float y = yCent - j;
              float x = (r - y * s) / c;
              PintarPixel (rgb, w, h, (int) lroundf (x + xCent), j);
            }
        }
      nLineas++;
    }
  free (cand);   // malloc -> free

  // --- Escritura del PNG ---
  // Se crea la carpeta "output"; si ya existe, mkdir falla con EEXIST y se
  // ignora.
  if (mkdir ("output", 0755) != 0 && errno != EEXIST)
    fprintf (stderr, "DrawLines: no se pudo crear la carpeta output\n");

  int ok = stbi_write_png (fname, w, h, 3, rgb, w * 3);
  free (rgb);   // malloc -> free

  if (!ok)
    {
      fprintf (stderr, "DrawLines: no se pudo escribir %s\n", fname);
      return -1;
    }

  printf ("Imagen de salida: %s (%d lineas de %d candidatas, threshold = %.1f votos)\n",
          fname, nLineas, nCand, threshold);
  return nLineas;
}
