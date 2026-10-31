/*
 ============================================================================
 Archivo     : pgm.cpp
 Autor       : Mario Rocha - Proyecto 3, Computacion Paralela, UVG
 Descripcion : Implementacion de PGMImage (ver pgm.h).
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "pgm.h"

/*****************************************************************
 * Funcion    : skipWhitespaceAndComments
 * Descripcion: Salta espacios en blanco y comentarios (lineas que
 *              inician con '#') del encabezado PGM.
 * Parametros : f - archivo abierto en modo lectura
 * Retorno    : ninguno (deja f en el siguiente caracter util)
 *****************************************************************/
static void skipWhitespaceAndComments (FILE *f)
{
  int c;
  while ((c = fgetc (f)) != EOF)
    {
      if (c == '#')
        {
          while ((c = fgetc (f)) != EOF && c != '\n')
            ;
        }
      else if (!isspace (c))
        {
          ungetc (c, f);
          return;
        }
    }
}

/*****************************************************************
 * Funcion    : readHeaderInt
 * Descripcion: Lee el siguiente entero del encabezado PGM. Termina
 *              el programa si no encuentra un entero valido.
 * Parametros : f - archivo abierto en modo lectura
 * Retorno    : el entero leido
 *****************************************************************/
static int readHeaderInt (FILE *f)
{
  int value;
  skipWhitespaceAndComments (f);
  if (fscanf (f, "%d", &value) != 1)
    {
      fprintf (stderr, "PGM: encabezado invalido\n");
      exit (1);
    }
  return value;
}

/*****************************************************************
 * Funcion    : PGMImage::PGMImage (constructor desde archivo)
 * Descripcion: Lee el encabezado (magic number, ancho, alto, maximo
 *              de gris) y luego los pixeles. Termina el programa si
 *              el archivo no existe o no es un PGM valido.
 * Parametros : fname - ruta del archivo .pgm
 * Retorno    : ninguno
 *****************************************************************/
PGMImage::PGMImage (const char *fname)
{
  FILE *f = fopen (fname, "rb");
  if (f == NULL)
    {
      fprintf (stderr, "PGM: no se pudo abrir '%s'\n", fname);
      exit (1);
    }

  // --- Encabezado ---
  char magic[3] = {0};
  if (fread (magic, 1, 2, f) != 2 || magic[0] != 'P' || (magic[1] != '5' && magic[1] != '2'))
    {
      fprintf (stderr, "PGM: '%s' no es un archivo P5/P2\n", fname);
      exit (1);
    }

  x_dim = readHeaderInt (f);
  y_dim = readHeaderInt (f);
  num_colors = readHeaderInt (f);
  if (x_dim <= 0 || y_dim <= 0 || num_colors <= 0 || num_colors > 255)
    {
      fprintf (stderr, "PGM: dimensiones o profundidad no soportadas\n");
      exit (1);
    }

  // --- Pixeles ---
  int total = x_dim * y_dim;
  pixels = new unsigned char[total];

  if (magic[1] == '5')
    {
      fgetc (f); // un solo caracter de espacio separa el encabezado de los datos
      if ((int) fread (pixels, 1, total, f) != total)
        {
          fprintf (stderr, "PGM: datos incompletos en '%s'\n", fname);
          exit (1);
        }
    }
  else
    {
      for (int i = 0; i < total; i++)
        pixels[i] = (unsigned char) readHeaderInt (f);
    }

  fclose (f);
}

/*****************************************************************
 * Funcion    : PGMImage::PGMImage (constructor de imagen vacia)
 * Descripcion: Crea una imagen de x * y pixeles en negro.
 * Parametros : x   - ancho
 *              y   - alto
 *              col - valor maximo de gris
 * Retorno    : ninguno
 *****************************************************************/
PGMImage::PGMImage (int x, int y, int col)
{
  x_dim = x;
  y_dim = y;
  num_colors = col;
  pixels = new unsigned char[x * y];
  memset (pixels, 0, x * y);
}

/*****************************************************************
 * Funcion    : PGMImage::~PGMImage
 * Descripcion: Libera el arreglo de pixeles (reservado con new[]).
 * Parametros : ninguno
 * Retorno    : ninguno
 *****************************************************************/
PGMImage::~PGMImage ()
{
  delete[] pixels;
}

/*****************************************************************
 * Funcion    : PGMImage::write
 * Descripcion: Guarda la imagen en formato P5 (binario).
 * Parametros : fname - ruta del archivo de salida
 * Retorno    : true si se escribieron todos los pixeles
 *****************************************************************/
bool PGMImage::write (const char *fname)
{
  FILE *f = fopen (fname, "wb");
  if (f == NULL)
    return false;

  fprintf (f, "P5\n%d %d\n%d\n", x_dim, y_dim, num_colors);
  int total = x_dim * y_dim;
  bool ok = (int) fwrite (pixels, 1, total, f) == total;
  fclose (f);
  return ok;
}
