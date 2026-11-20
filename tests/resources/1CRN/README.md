# 1CRN

[Crambin (PDB 1CRN)](https://www.rcsb.org/structure/1CRN) contains three disulfide bonds: Cys3–Cys40, Cys4–Cys32, and Cys16–Cys26.

Atomworks excludes disulfide connections by default. Preserve them with `add_bond_types_from_struct_conn`:

```python
import atomworks

structure = atomworks.parse(
    filename=cif_path,  # Path to the 1CRN CIF file.
    add_bond_types_from_struct_conn=["covale", "disulf"],
)
```
