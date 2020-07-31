#!/bin/bash

# 1. Download PNG*.zip and cube+_*.txt from https://ipg.fer.hr/ipg/resources/color_constancy (link to Cube+)
# 2. Download *.zip from https://www.isispa.org/illumination-estimation-challenge/data
# 3. Copy downloaded files to cc_data/cube/
# 4. Run this script

# Cube+
mkdir plus
mkdir plus/gt
cp cube+_gt.txt plus/gt/gt.txt
cp cube+_left_gt.txt plus/gt/left_gt.txt
cp cube+_right_gt.txt plus/gt/right_gt.txt

# Iterate the string array using for loop
for file in `ls PNG*.zip`; do
   unzip ${file} -d PNG
done

for file in `ls PNG/*.PNG | xargs -n 1 basename`; do
  mv PNG/$file PNG/${file/PNG/png}
done

mv PNG plus/

# Cube ISPA 2019 challenge
mkdir challenge
mkdir challenge/gt

unzip -P \*^\$5CwRxr\#b\?nq\#TnD2Wtw5F test_gt.zip
mv test_gt.txt challenge/gt/gt.txt

unzip -P kEt2xPR\?^%rqGKDj4LR8Z\*NK images.zip -d PNG
mv PNG challenge/
