#!/bin/bash
#change to the directory
cd ../1.SurfaceGeneration/
echo the first argument is the code of pdb in lowercase
echo the second argument is the file address of the pdb
#run the python script
python checkUnknownAA.py $1 $2


