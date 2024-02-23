A. Dependency
>torch<br>
>pykeops<br>
>numpy<br>
>sklearn<br>

B. Preparation for proteins
>1.Download the .pdb file locally<br>
>2.Using Reduce (https://github.com/rlabduke/reduce) for adding and correcting hydrogens in PDB files<br>
>3.Using APBS (https://server.poissonboltzmann.org/) to get the continuum electrostatics.<br>

C.Protein surface generation and feature calculation.
>4.Run checkUnknownAA.py to check if there are any unknow AA.If yes, edit them into the dictionary.py.Else to the next step. This is important to generate the SES<br>
