/*
 ============================================================================
 Archivo     : pgm.h
 Descripcion : Lectura y escritura de imagenes PGM en escala de grises
               (formatos P5 binario y P2 ASCII, 8 bits por pixel).
 ============================================================================
 */
#ifndef PGM_H
#define PGM_H

class PGMImage
{
public:
  int x_dim;              // ancho de la imagen (w)
  int y_dim;              // alto de la imagen (h)
  int num_colors;         // valor maximo de gris (normalmente 255)
  unsigned char *pixels;  // x_dim * y_dim pixeles, fila por fila

  // Carga la imagen desde un archivo .pgm. Termina el programa si falla.
  PGMImage (const char *fname);

  // Crea una imagen vacia (en negro) de x * y pixeles.
  PGMImage (int x, int y, int col = 255);

  ~PGMImage ();

  // Guarda la imagen en formato P5. Devuelve true si tuvo exito.
  bool write (const char *fname);
};

#endif
