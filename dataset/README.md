# Dataset Descriptions
## LIGYSIS Human Binding Sites — April 2024

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

## LIGYSIS_human_chains_per_lig_MAY_2024
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
[1]J. Sánchez Utgés, 《LBS-Comparison results》. Zenodo, 1月 21, 2025. doi: 10.5281/zenodo.14645504.
