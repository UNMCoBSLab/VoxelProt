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

