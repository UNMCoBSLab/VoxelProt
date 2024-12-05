#!/bin/bash
#change to the directory
cd ../1.SurfaceGeneration/
echo the first argument is the code of pdb in lowercase
echo the second argument is the file address of the pdb
echo the third argument is the .dx file address
echo the fourth argument is the directory to store all features.
#create the directory
mkdir -p $4/atomtype/
mkdir -p $4/candidates/
mkdir -p $4/dictionary/
mkdir -p $4/electro_info/
mkdir -p $4/KH/
mkdir -p $4/shape_index3/
mkdir -p $4/shape_index/
mkdir -p $4/surfaceNormals/
#run the python script
python feature2CSV_v3.py $1 $2 $3 $4
