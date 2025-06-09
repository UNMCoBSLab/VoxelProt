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
>7.train the model<br>
./trainmodel.sh trainsetList(csv) directoryWhereBindingSiteOctreeAreStored directoryWhereNonBindingSiteOctreeAreStored directoryWhereTrainedModelWillBeStored numOfEpoches<br>
e.g. bash trainmodel.sh /../example_protein_list.csv /.../bindingsite/ /.../nonbindingsite/ /.../VoxelProt/voxelProtmodel 2<br>


F.Detection cofactor-binding site
>7.detect the cofactor-binding site using the well-trained model in E<br>
./detection.sh codeOfPdb fileAddressOfThePDB fileAddressOfTheTrainedModel addressToStoreFeatures outputname<br>
e.g. bash detection.sh 1a27 /../pdb1a27.ent /../voxelProtmodel1.pth /../features/ /../bindingsite1a27.csv'<br>
