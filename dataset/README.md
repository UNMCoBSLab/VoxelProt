# Dataset Descriptions
## 27 cofactors in the CoFactor database[1]

| #  | Cofactor (Name + Abbreviation)                     | PDB Ligand ID |
|----|----------------------------------------------------|----------------|
| 1  | Ascorbate (Ascorbic acid)                          | ASC            |
| 2  | Adenosyl-cobalamin (Vitamin B12)                   | ADO, CBL       |
| 3  | Biopterin                                           | BPT            |
| 4  | Biotin                                              | BTN            |
| 5  | Coenzyme A (CoA)                                    | COA            |
| 6  | Coenzyme B (CoB)                                    | COB            |
| 7  | Coenzyme M (CoM)                                    | COM            |
| 8  | Coenzyme Q (Ubiquinone, CoQ)                        | UQ1, UQ2, UQ9   |
| 9  | Dipyrro-methane                                     | Not assigned   |
| 10 | Factor 430 (Cofactor F430)                          | F43            |
| 11 | Flavin adenosine diphosphate (FAD)                 | FAD            |
| 12 | Flavin mono-nucleotide (FMN)                        | FMN            |
| 13 | Glutathione                                         | GSH            |
| 14 | Heme                                                | HEM            |
| 15 | Lipoic acid                                         | LPA, LIO       |
| 16 | MIO (4-methyldieneimidazole-5-one)                  | MIO            |
| 17 | Molybdenum cofactor (Molybdopterin, MoCo)           | MPT, MOO       |
| 18 | Nucleotidylated MoCo (R = GMP or CMP)               | GTP/CMP-MPT    |
| 19 | Menaquinone (Vitamin K2)                            | MQN, MEN       |
| 20 | Nicotinamide adenine dinucleotide (NAD, NADP)       | NAD, NAP       |
| 21 | Pyridoxal-phosphate (PLP)                           | PLP            |
| 22 | Phosphopantetheine                                   | PPN, PPT       |
| 23 | Pyrroloquinoline quinone (PQQ)                      | PQQ            |
| 24 | S-adenosyl-methionine (SAM, AdoMet)                 | SAM            |
| 25 | Thiamine diphosphate (ThDP, TDP)                    | TPP, THD       |
| 26 | Tetrahydrofolic acid (Folic acid, Folate)           | FOL, THG       |
| 27 | Topaquinone (TPQ)                                   | TPQ            |
|    | Ortho-quinones (TTQ, LTQ, CTQ)                      | TTQ, CTQ, LTQ  |


## LIGYSIS Human Binding Sites — April 2024[2]

It is part of the **LIGYSIS** resource, which provides curated annotations of protein–ligand binding sites across the **human proteome**. 

### Description

| Column      | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `ACC`       | UniProt accession ID of the protein (e.g., `P12345`)                        |
| `SEG`       | Segment index (typically `1`; higher values indicate domain splits)         |
| `ID`        | Binding site index within the segment (e.g., `0`, `1`, `2`, ...)            |
| `up_aas`    | List of 1-based UniProt residue indices involved in the binding site        |
| `n_up_aas`  | Number of residues listed in `up_aas`                                       |
| `SEG_ID`    | Concatenated string `{ACC}_{SEG}` used as a key (e.g., `P12345_1`)          |
| `SITE_NAME` | Globally unique name for the site, formatted as `{SEG_ID}_{ID}` (e.g., `P12345_1_0`) |


### Example

| ACC     | SEG | ID | up_aas                                      | n_up_aas | SEG_ID     | SITE_NAME     |
|---------|-----|----|---------------------------------------------|----------|------------|----------------|
| P41182  | 1   | 0  | [11, 13, 14, 17, 18, 21, 22, ..., 30, ...]   | 33       | P41182_1   | P41182_1_0     |


This entry describes a ligand-binding site located on the first segment (`SEG = 1`) of protein `P41182`, and it is the first site (`ID = 0`) in that segment. The binding site spans **33 UniProt residues**, and it is uniquely identified across datasets using the key `P41182_1_0`.

You can use this `SITE_NAME` to look up matching ligands or 3D structures in other associated LIGYSIS datasets ('LIGYSIS_human_chains_per_lig_MAY_2024.pkl').



## LIGYSIS_human_chains_per_lig_MAY_2024[2]
It is part of the **LIGYSIS** resource and provides mappings between human protein segments and **ligand-bound structural binding sites**. 
### Description

```python
{
  "<UniProt_ID>_<Isoform>": {
    <site_index>: {
      "<PDB>_<Ligand>_<Chain>_<ResidueIndex>": {
        set of interacting protein chain IDs (e.g., {'A', 'B'})
      }
    }
  }
}
```
### Example
```python
{
  "A0AVT1_1": {
    0: {
      "7sol_IHP_E_1101": {"A"}
    },
    1: {
      "7sol_AMP_F_1101": {"A", "B"},
      "7pvn_ATP_C_1101": {"A"}
    },
    2: {
      "7pvn_CA_I_1107": {"A"}
    },
    3: {
      "7pvn_CA_J_1108": {"A"}
    }
  }
}
```
| Site | Ligand Record       | Ligand Info                                    | Interacting Chains |
|------|---------------------|------------------------------------------------|---------------------|
| 0    | 7sol_IHP_E_1101     | IHP in PDB 7sol, Chain E, Residue 1101         | A                   |
| 1    | 7sol_AMP_F_1101     | AMP in 7sol, Chain F, Residue 1101             | A, B                |
| 1    | 7pvn_ATP_C_1101     | ATP in 7pvn, Chain C, Residue 1101             | A                   |
| 2    | 7pvn_CA_I_1107      | Calcium ion in 7pvn, Chain I, Residue 1107     | A                   |
| 3    | 7pvn_CA_J_1108      | Calcium ion in 7pvn, Chain J, Residue 1108     | A                   |

### Ref:
[1] Fischer, J. D., Holliday, G. L., & Thornton, J. M. (2010). The CoFactor database: organic cofactors in enzyme catalysis. Bioinformatics, 26(19), 2496-2497.
[2] J. Sánchez Utgés, 《LBS-Comparison results》. Zenodo, 1月 21, 2025. doi: 10.5281/zenodo.14645504.
