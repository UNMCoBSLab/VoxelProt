You can run a single test case using detection_example.py.

To reproduce all experiments, follow the steps below in order.
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
>5.run ../detection/detection_runner.py to get all predicted binding sites<br>

G.Evaluation
>6.run ../evaluation/get_true_binding_site_runner.py to get all true binding site and store them in .pdb<br>
>7.run ../evaluation/voxelprot_runner.py to get all true binding site and store them in .pdb<br>

F.fpocket
>1. install and run fpocket
>2. run ../evaluation/fpocket_runner.py to get the results of fpocket

G.P2Rank
>1. install and run P2Rank
>2. run ../evaluation/prank_runner.py to get the results of P2Rank

H.DeepSurf
>1. install and run DeepSurf
>2. run ../evaluation/deepsurf_runner.py to get the results of DeepSurf

H.Kalasanty
>1. install and install Kalasanty
>2. run ../evaluation/kalasanty_runner.py to get the results of Kalasanty
