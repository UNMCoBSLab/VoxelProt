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
>4.Run checkUnknownAA.py to check if there are any unknow AA.If yes, edit them into the dictionary.py.Else to the next step. This is important to generate the SES<br>
>5.Run SESFeature2CSV.py to generate SES and assign features to each surface points<br>

D.Creating Octree objects.
>6.Run surface2boxes_SEScg2e.py to create and store Octree objects<br>

