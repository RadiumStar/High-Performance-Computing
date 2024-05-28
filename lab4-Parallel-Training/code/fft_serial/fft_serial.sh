#! /bin/bash
#
gcc -c -Wall fft_serial.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi
#
gcc fft_serial.o -lm
if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi
#
rm fft_serial.o
#
chmod ugo+x a.out
mv a.out ~/binc/fft_serial
#
echo "Normal end of execution."
