#!/bin/bash
#change to the directory
cd VoxelProt/1.SurfaceGeneration/
echo the first argument is the code of pdb in lowercase
echo the second argument is the file address of the pdb
echo the third argument is the .dx file address
echo the fourth argument is the directory to store all features.
#run the python script
python feature2CSV_v3.py $1 $2 $3 $4
