#!/usr/bin/env bash
# ============================================================================
# run_bitacora.sh
# Corre ./hough varias veces y guarda el tiempo del kernel de cada corrida.
#
# Uso: ./run_bitacora.sh <version> [corridas] [imagen]
#   version  : kernel a correr y nombre del CSV (global, constante)
#   corridas : numero de ejecuciones (default 10)
#   imagen   : imagen PGM de entrada (default runway.pgm)
#
# Salida: bitacora/<version>.csv  (corrida,tiempo_ms)
#         y en pantalla el promedio y la desviacion estandar.
# ============================================================================
set -euo pipefail

VERSION="${1:?Uso: $0 <version> [corridas] [imagen]}"
RUNS="${2:-10}"
IMG="${3:-runway.pgm}"
OUT="bitacora/${VERSION}.csv"

make -s hough
mkdir -p bitacora
echo "corrida,tiempo_ms" > "$OUT"

for i in $(seq 1 "$RUNS"); do
  salida=$(./hough "$IMG" "$VERSION")

  if echo "$salida" | grep -q "mismatch"; then
    echo "Aviso: la corrida $i tuvo diferencias contra la version CPU" >&2
  fi

  t=$(echo "$salida" | awk '/Tiempo kernel:/ {print $3}')
  if [ -z "$t" ]; then
    echo "Error: no se encontro el tiempo en la corrida $i" >&2
    exit 1
  fi

  echo "$i,$t" >> "$OUT"
  echo "Corrida $i: $t ms"
done

# promedio y desviacion estandar muestral (n - 1)
awk -F, 'NR > 1 { n++; s += $2; ss += $2 * $2 }
  END {
    prom = s / n
    de = (n > 1) ? sqrt((ss - n * prom * prom) / (n - 1)) : 0
    printf "\nCorridas: %d\nPromedio: %f ms\nDesv. estandar: %f ms\n", n, prom, de
  }' "$OUT"

echo "Guardado en $OUT"
