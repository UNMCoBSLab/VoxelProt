#!/bin/bash
#change to the directory
cd ../2.Surface2Octree/
echo the first argument is the code of the pdb in lowercas
echo the second argument is the file address of the pdb
echo the third argument is the feature address
echo the fourth argument is the directory to store the binding site octrees
echo the fifth argument is the directory to store the non-binding site octrees
mkdir -p $4
mkdir -p $5
python surface2boxes-github.py $1 $2 $3 $4 $5
