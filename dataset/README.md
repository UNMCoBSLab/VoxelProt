# LIGYSIS Human Binding Sites — April 2024

`LIGYSIS_human_sites_APRIL_2024.pkl` is part of the **LIGYSIS** resource, which provides curated annotations of protein–ligand binding sites across the **human proteome**. 

## Description

| Column      | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `ACC`       | UniProt accession ID of the protein (e.g., `P12345`)                        |
| `SEG`       | Segment index (typically `1`; higher values indicate domain splits)         |
| `ID`        | Binding site index within the segment (e.g., `0`, `1`, `2`, ...)            |
| `up_aas`    | List of 1-based UniProt residue indices involved in the binding site        |
| `n_up_aas`  | Number of residues listed in `up_aas`                                       |
| `SEG_ID`    | Concatenated string `{ACC}_{SEG}` used as a key (e.g., `P12345_1`)          |
| `SITE_NAME` | Globally unique name for the site, formatted as `{SEG_ID}_{ID}` (e.g., `P12345_1_0`) |


## Example

```python
import pickle
import pandas as pd

with open("LIGYSIS_human_sites_APRIL_2024.pkl", "rb") as f:
    df = pickle.load(f)

print(df.head())
