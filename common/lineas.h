/*
 ============================================================================
 Archivo     : lineas.h
 Autor       : Luis Pedro Lira - Proyecto 3, Computacion Paralela, UVG
 Descripcion : Dibujo de las lineas detectadas por la transformada de Hough
               sobre la imagen de entrada y escritura del resultado en PNG.
 ============================================================================
 */
#ifndef LINEAS_H
#define LINEAS_H

/*****************************************************************
 * Funcion    : DrawLines
 * Descripcion: Selecciona las celdas (r, theta) del acumulador con mas
 *              votos y dibuja en rojo sus rectas sobre una copia RGB de
 *              la imagen de entrada. Guarda el resultado en un PNG.
 *              Una celda se dibuja si:
 *                1) sus votos superan threshold = promedio + 2 * desv.
 *                   estandar de todas las celdas del acumulador, y
 *                2) es un maximo local: ninguna de sus 8 celdas vecinas
 *                   tiene mas votos (evita dibujar muchas rectas casi
 *                   iguales alrededor de la misma linea real).
 * Parametros : fname      - ruta del PNG de salida (se crea la carpeta
 *                           "output" si no existe)
 *              pic        - pixeles de la imagen en gris (w * h, fila por
 *                           fila)
 *              w, h       - ancho y alto de la imagen
 *              acc        - acumulador de rBins * degreeBins celdas, con el
 *                           indice rIdx * degreeBins + tIdx
 *              degreeBins - numero de bins de theta
 *              rBins      - numero de bins de r
 *              radInc     - incremento de theta en radianes
 *              rMax       - radio maximo (centro -> esquina)
 *              rScale     - ancho de cada bin de r
 * Retorno    : numero de lineas dibujadas, o -1 si no se pudo guardar
 *****************************************************************/
int DrawLines (const char *fname, unsigned char *pic, int w, int h,
               int *acc, int degreeBins, int rBins, float radInc,
               float rMax, float rScale);

#endif
