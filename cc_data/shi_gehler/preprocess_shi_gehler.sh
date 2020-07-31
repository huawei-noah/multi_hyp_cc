#!/bin/bash

# 1. Download png_*.zip from https://www2.cs.sfu.ca/~colour/data/shi_gehler/
# 3. Copy downloaded files to cc_data/shi_gehler/
# 4. Run this script

# Iterate the string array using for loop
for file in `ls png_*.zip`; do
   unzip ${file}
done

mv cs/chroma/data/canon_dataset/568_dataset/png ./images
rm -rf cs
