# predictions.json transfer packet

The prediction outputs omitted from the checkpoint packet are bundled in `runs/pc_predictions_v1.tar.gz`.

- 24 original `predictions.json` files from E35/E37/E38 parents and branches
- 1,020,858,798 uncompressed bytes
- Archive: 26,088,641 bytes
- Archive SHA-256: `482e168c2e7a295ab2b7acc6ad0d5fc7d2a20864bb4a4451d48dbd59c251cc59`
- Inner `PREDICTIONS_MANIFEST.json` records each original path, byte count, and SHA-256

Upload this archive to the AI2 task. After extraction, verify `SHA256SUMS` and the manifest before using any rows for the CPU migration gate. The archive contains predictions only; it does not authorize training or replace the checkpoint lineage packet.
