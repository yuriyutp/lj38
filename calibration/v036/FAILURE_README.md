# v0.3.6 calibration failure record

The preregistered calibration stopped before Contract B because one pilot PT cell retained 39 observations while the frozen minimum was 40. No threshold was changed after observing the result, and no `CONTRACT_B_V036.json`, `MANIFEST_B_V036.json`, or anchor-3 was issued.

The full local raw ledger is 77,549,658 bytes. Its checkpoint is 137,042,731 bytes and exceeds GitHub's ordinary single-file limit. They are therefore not committed here. `CALIBRATION_FAILURE_V036.json` records their measured SHA-256 digests, sizes, row/key counts, duplicate count, and canonical equality so a separately transferred copy can be verified.

`CALIBRATION_INTERRUPTION_V036_001.json` records a conservative manual interruption and the exact atomic-checkpoint recovery. It did not change the final 37-row ledger or the scientific status, which remains `UNVERIFIED`.
