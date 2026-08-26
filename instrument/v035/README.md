# LJ38 audit instrument v0.3.5

This directory contains the instrument frozen for anchor-2 of
`lj38-audit-v0.3.5-contractA`.

## Contents

- `md_search_v035.py`: A0-A7 and B1/B2, exact PES budget accounting,
  `Q_inst`, history/yoke rules, diagnostics, and raw/checkpoint helpers.
- `crosscheck_v035.py`: uploaded cross-check runner, retained unchanged as
  audit history.
- `crosscheck_v035_anchor2.py`: executable runner with pre-anchor corrections:
  S_38 descriptor testing, active-confinement COM derivative reference,
  `incomplete_quenches` as a pass condition, reference-file digests, and
  exact NumPy/SciPy preflight.
- `refs/LJ38_fcc.txt`: Cambridge Cluster Database `points/38`.
- `refs/LJ38_ico.txt`: Cambridge Cluster Database `points/38i`.

The reference coordinate sources are:

- https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/38
- https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/points/38i

B3 minima hopping is defined in Contract A but is not a promotion target in
this round and is not implemented by this anchor-2 instrument.

Run the cross-check with the pinned environment:

```text
python crosscheck_v035_anchor2.py --instrument md_search_v035 \
  --path . --repo <repository-root> \
  --ref-fcc refs/LJ38_fcc.txt --ref-ico refs/LJ38_ico.txt \
  --out INSTRUMENT_MANIFEST_V035.json
```
