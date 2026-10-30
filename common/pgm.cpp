/*
 ============================================================================
 Archivo     : pgm.cpp
 Descripcion : Implementacion de PGMImage (ver pgm.h).
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "pgm.h"

//*****************************************************************
// Salta espacios en blanco y comentarios (lineas que inician con '#')
// del encabezado PGM.
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

//*****************************************************************
// Lee el siguiente entero del encabezado PGM.
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

//*****************************************************************
PGMImage::PGMImage (const char *fname)
{
  FILE *f = fopen (fname, "rb");
  if (f == NULL)
    {
      fprintf (stderr, "PGM: no se pudo abrir '%s'\n", fname);
      exit (1);
    }

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

//*****************************************************************
PGMImage::PGMImage (int x, int y, int col)
{
  x_dim = x;
  y_dim = y;
  num_colors = col;
  pixels = new unsigned char[x * y];
  memset (pixels, 0, x * y);
}

//*****************************************************************
PGMImage::~PGMImage ()
{
  delete[] pixels;
}

//*****************************************************************
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
