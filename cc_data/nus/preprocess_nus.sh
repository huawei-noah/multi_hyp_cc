#!/bin/bash

# 1. Download MASK, GROUNDTRUTH and PNG from http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html
# 2. Copy all files downloaded to cc_data/nus/
# 3. Run this script

declare -a CAMERAS=("Canon1DsMkIII" "Canon600D" "FujifilmXM1" "NikonD5200" "PanasonicGX1" "OlympusEPL6" "SonyA57" "SamsungNX2000")

# Iterate the string array using for loop
for camera in ${CAMERAS[@]}; do
   mkdir $camera
   cat ${camera}_PNG.zip.00* > ${camera}_PNG.zip
   unzip ${camera}_PNG.zip
   #rm  ${camera}_PNG.zip.00*
   rm ${camera}_PNG.zip
   mv PNG $camera/

   mkdir $camera/gt/
   cp ${camera}_gt.mat $camera/gt/gt.mat
   #rm ${camera}_gt.mat

   unzip ${camera}_CHECKER.zip
   #rm ${camera}_CHECKER.zip
   mv CHECKER $camera/mask
done
