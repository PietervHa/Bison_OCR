# Development

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyflakes httpx      # dev only: lint + the web tests
```

Run everything from the project root. Config paths such as `data/results` and `debug_dir` are resolved relative to the project root, not the working directory.

## Checks

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m pyflakes backend frontend QC_tools tests
```

| Test module | Covers |
|---|---|
| `test_font_reader.py` | font reader on real fixture crops (rib through a 6, crossing ribs, upside-down cap), a verification model, the OCRead result/reject rules, font-reader-unavailable handling |
| `test_code_validator.py` | template fielding: never invents a digit, look-alike fixes, value_pattern |
| `test_dot_text.py` | dot-text cleanup keeps ink on a rib and removes the rib |
| `test_ocread_deskew.py` | text locator and deskew orientation |
| `test_integration_vision_pipeline.py` | `run_vision` status rules (sync and callback), JSONL writer |
| `test_tcp_trigger_ocread.py` | PLC response: code text, line breaks, length limit, NOK |
| `test_web_controls.py` | camera rotation and counter reset work without login; no maintenance route (needs httpx) |
| `test_camera_device_lookup.py`, `test_camera_recovery.py` | device lookup by name, reconnect and frozen-frame handling (fake camera) |
| `test_qc_tools.py` | session report, project-relative paths |
| `test_unit_core.py` | AppState counters and snapshot |

`tests/camera_test.py` is a manual script that shows the live camera in a window (`python -m tests.camera_test`). It is not a unit test.

## QC tools

| Tool | Use |
|---|---|
| `python -m QC_tools.q_trigger` | Sends TCP triggers to the running app like the PLC (`--interval`, `--max-triggers`, `--hours`, `--host`, `--port`; `--mode keyboard` simulates the Q key) |
| `python -m QC_tools.ocread_session_report` | Summary of a results file: OK/NOK, reject reasons, timings, with `--expected` correct/wrong/rejected (`--since`, `--until`, `--last`) |
| `python -m QC_tools.camera_stability` | Measures brightness drift to check that `camera.manual_settings` really pinned exposure |

## Debug captures

Set `ocr.dynamic_roi.debug_save: true`. Each read then writes `data/debug/dynamic_roi/<timestamp>/`:

- `text_locator_*.png`: search region, ink masks, located box
- `crop_raw_preprocess.png`: the crop handed to the font reader
- `font_reader.txt`: the read summary. For a rejected read it also holds the rejected decode, which is never shown on the HMI or sent to the PLC.

This writes files on every read, so leave it off in production.

## Logs

`data/logs/vision.log` (DEBUG and up) and the console (INFO and up). Routine `/result` polling requests are filtered out. Every read logs a `VISION RESULT:` line with status, reason and cycle time.

## Changing the print format

The font reader reads two lines of digits plus the literals `B`, `:` and `/` from the current template. For a new format, change `ocr.read_validation.template` and `value_pattern`. If the font reader does not support the new template, the app logs an error at startup and every read is NOK with `ocr_engine_error`. The font reader then has to be retrained, which is not part of this repository.

## Rules worth keeping

- A rejected read never carries a code: `text` is empty and `detections` is empty.
- A technical failure is reported as `ocr_engine_error`, never as "no text".
- The text sent to the PLC never contains a line break, and an over-long read is NOK rather than truncated.
