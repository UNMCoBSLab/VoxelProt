#!/bin/bash
#change to the directory
cd ../3.trainModel/
echo the first argument is a list of proteins
echo the second argument is the directory where binding site Octrees are stored
echo the third argument is the directory where non-binding site Octrees are stored
echo the fourth argument is where the trained model will be stored
echo the fifth argument is the number of epoches
python trainmodel.py $1 $2 $3 $4 $5
