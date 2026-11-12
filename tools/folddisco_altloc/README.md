# folddisco-altloc

How well is each conformer of a loop, together with its structural context, represented
in the structure databases? Give this tool a PDB file in which a loop is modeled with two
alternate conformers (altlocs). It writes one structure per conformer, picks a small motif
that tells the two apart, searches both with the
[FoldDisco webserver](https://search.foldseek.com/folddisco), and reports how many close
matches each conformer has in each database.

## Install

Python 3.10 or newer. You need read access to the `prism-science/sampleworks` repository.
Install with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv tool install "git+ssh://git@github.com/prism-science/sampleworks.git@mdc/folddisco#subdirectory=tools/folddisco_altloc"
```

```bash
pip install "git+ssh://git@github.com/prism-science/sampleworks.git@mdc/folddisco#subdirectory=tools/folddisco_altloc"
```

## Use the web interface

```bash
folddisco-altloc serve
```

This opens `http://localhost:8417/` in your browser (the next free port if 8417 is taken).
The page and server run only on your machine; searches go to search.foldseek.com.

1. **Drop a PDB file** onto the page. It lists every run of residues that has altlocs and
   preselects the longest one with exactly two complete conformers.
2. **Pick the residues**: click a segment, or type a chain and range such as `158-167`.
3. **Set the options** (below) and press **Search**. A search usually takes under a minute.
4. **Read the table**: for each database, the number of target structures with a close match
   to conformer A and to B, and how many matched both. Download the summary as CSV, or every
   file of the run as a zip.

Runs are saved under `~/folddisco-altloc-runs/` (change with `--workdir`).

## Options

| Option | Default | What it does |
|---|---|---|
| Motif residues | 4 | Total motif size, split evenly: residues from your loop that move most between the conformers, plus the same number of *anchors* (residues without altlocs whose distances to the moving residues change most). |
| Count hits matching at least | all | A hit counts only if it matches this many motif residues. |
| RMSD threshold | 0.5 Å | A hit counts only if its RMSD is below this. |
| Amino acid substitution | none | `group` lets each motif residue match its class (RHK, DE, NQST, FWY, AVLIMC, GP), like `folddisco --aa-subst group`. `size` and `blosum62` are also available. |
| Confident hits only | off | Also require FoldDisco's `--confident` rule: at least 80% of motif residues matched within 1 Å. |
| Databases | all five | PDB, AlphaFold DB 50, AlphaFold DB proteomes, ESM Atlas 30, BFVD. |
| Distance cutoff (advanced) | 12 Å | Moving–anchor pairs farther apart than this in both conformers are ignored. |
| Min. sequence separation (advanced) | 3 | Anchors must be at least this far in sequence from your residue set. |

## Command line

The same pipeline without a browser, printing the table:

```bash
folddisco-altloc run 5MHX.pdb --chain A --residues 158-167 --motif-size 4 --aa-subst group --confident --rmsd-max 0.5
```

Also available: `folddisco-altloc motif` (write the conformer PDBs and FoldDisco query file
only) and `folddisco-altloc search` (run a FoldDisco batch query file). Use `--help` on any
of them.

## Reading the results

- **Targets (distinct numberings).** Targets are unique structures. The number in
  parentheses counts distinct matched residue numberings, a rough proxy for distinct
  proteins: many PDB entries of one protein usually match the same residue numbers.
- **`*` — possibly cut off.** The webserver returns at most 1,000 hits per database, ranked
  by FoldDisco's `match_score` (IDF × coverage² × TM-score), not by RMSD. A close match can
  rank below hundreds of partial matches. When counted hits reach the last 20% of a full
  list, more were probably dropped; treat that count as a lower bound.
- **Altlocs of other entries are not searched.** FoldDisco ignores altloc labels and keeps
  the last conformer listed for each residue, so it indexes one conformer per deposited
  structure. Your own query structure is in the PDB index in its last-listed conformer
  only. Anchors are restricted to residues without altlocs for the same reason.
- **`--aa-subst` and `--confident` are applied by this tool.** The webserver accepts no
  extra flags, so substitutions are written into the motif (`A181` becomes `A181:DE`) and
  the confident rule is applied to the downloaded hits. Both follow FoldDisco's documented
  definitions.

## Limitations

- PDB format only (`.pdb`, `.ent`); convert mmCIF first, for example with
  `gemmi convert in.cif out.pdb`.
- One chain per residue set; insertion codes are not supported.
- Exactly two altlocs are compared. Residues lacking either one must be left out of the
  range, otherwise they would be deleted from one conformer file.

## Development

```bash
cd tools/folddisco_altloc
uv venv && uv pip install -e ".[test]"
.venv/bin/python -m pytest tests
```

The tests run offline; nothing in them contacts the FoldDisco server.
