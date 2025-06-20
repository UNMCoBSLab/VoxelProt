A. Dependency
>torch<br>
>pykeops<br>
>numpy<br>
>sklearn<br>
>biopython<br>
>tqmd
>Reduce for adding and correcting hydrogens in PDB files(https://github.com/rlabduke/reduce)<br>
>Using APBS (https://server.poissonboltzmann.org/) to get the continuum electrostatics.<br>
>MDAnalysis (https://www.mdanalysis.org/GridDataFormats/gridData/overview.html).<br>

B. Preparation for proteins
>1.Run ../dataset/preprocess.py<br>


C.Protein surface generation and feature calculation.
>2.Run ../surfaceGeneration/SASGeneration_runner.py<br>

D.Creating Octree objects.
>3.Run ../surface2Octree/octree_generation_runner.py<br>


E.Training the model
>4.Run ../trainModel/train_model_runner.py<br>


F.Detection cofactor-binding site
>7.detect the cofactor-binding site using the well-trained model in E<br>
./detection.sh codeOfPdb fileAddressOfThePDB fileAddressOfTheTrainedModel addressToStoreFeatures outputname<br>
e.g. bash detection.sh 1a27 /../pdb1a27.ent /../voxelProtmodel1.pth /../features/ /../bindingsite1a27.csv'<br>

G.Evaluation
>6.run ../evaluation/get_true_binding_site_runner.py to get all true binding site and store them in .pdb<br>
>7.run ../evaluation/voxelprot_runner.py to get all true binding site and store them in .pdb<br>

F.fpocket
>1. install fpocket
>2. for pdb in /path/split_proteins/*.pdb; do
    fpocket -f "$pdb"
    done
>3. run ../evaluation/fpocket_runner.py to get the results of fpocket

G.P2Rank
>1. install P2Rank
>2. for pdb in /path/split_proteins/*.pdb; do
    /path/p2rank_2.5/prank predict -f "$pdb" -o /path/p2rank_output/
    done 
>3. run ../evaluation/prank_runner.py to get the results of prank

