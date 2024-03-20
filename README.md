A. Dependency
>torch<br>
>pykeops<br>
>numpy<br>
>sklearn<br>
>biopython<br>

B. Preparation for proteins
>1.Download the .pdb file locally<br>
>2.Using Reduce (https://github.com/rlabduke/reduce) for adding and correcting hydrogens in PDB files<br>
>3.Using APBS (https://server.poissonboltzmann.org/) to get the continuum electrostatics.<br>

C.Protein surface generation and feature calculation.
>4.Check if there are any unknow AA.If yes, edit them into the dictionary.py.Else to the next step. <br>
chmod u+x checkUnknownAA.sh<br>
./ checkUnknownAA.sh codeOfPdb fileAddressOfThePDB<br>
e.g. ./checkUnknownAA.sh 1a27 /home/pdb1a27.ent<br>
>5.Generate the surface point cloud model and assign features to each surface points<br>
./ feature2CSV.sh codeOfPdb fileAddressOfThePDB DxFileAddress addressToStoreFeatures<br>
e.g. ./ feature2CSV.sh 1a27 /home/pdb1a27.ent /home/1a27.dx /home/features/<br>

D.Creating Octree objects.
>6.Create and store Octree objects<br>
./surface2Octree.sh codeOfPdb fileAddressOfThePDB addressStoreFeatures addressToStorebindingSiteOctree addressToStoreNonbindingSiteOctree<br>
e.g. ./surface2Octree.sh 1a27 /home/pdb1a27.ent /home/features/ /home/bindingsite/ /home/nonbindingsite/<br>

E.Training the model
>7.train the model<br>
./trainmodel.sh trainsetList(csv) directoryWhereBindingSiteOctreeAreStored directoryWhereNonBindingSiteOctreeAreStored directoryWhereTrainedModelWillBeStored numOfEpoches<br>
e.g. ./trainmodel.sh /home/proteinlist.csv /home/bindingsite/ /home/nonbindingsite/ /home/VoxelProt/voxelProtmodel 2<br>


F.Detection cofactor-binding site
>7.detect the cofactor-binding site using the well-trained model in E<br>
./detection.sh codeOfPdb fileAddressOfThePDB fileAddressOfTheTrainedModel addressToStoreFeatures outputname<br>
e.g. ./detection.sh 4p68 /home/pdb4p68.ent "model.pth" /home/features/ /home/bindingsite4p68.csv'<br>
