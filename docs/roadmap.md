# Roadmap

## Open

### Stability test on the line
Run the system on the production line for at least one full working day without manual intervention. Watch `data/logs/vision.log` for camera reconnects, `vision_timeout` and `ocr_engine_error`, and check memory use. Use `QC_tools/ocread_session_report.py` on that day's results file to see how often the code was read correctly, read wrong, or rejected.

### Font reader on new products
The font reader's leave-one-bottle-out check covers four bottles (4113, 3030, 5203, 5307). Before relying on it for a new product, check it with the verification models (see `docs/configuration.md`) and with `ocread_session_report --expected` on a session with that product.
