# Configuration

All settings are in `config/default.yaml`. Start with `--config <path>` to use another file. `default.yaml` itself has the detailed comments and the measurements behind the values. This page is the overview.

## Top level

| Key | Default | Meaning |
|---|---|---|
| `machine_id` | `"default"` | Station name shown in the HMI header |

## `ocr` – the OCRead pipeline

| Key | Default | Meaning |
|---|---|---|
| `downscale` | `0.6` | Shrink factor for the located crop before it is read |
| `min_dim` | `480` | Never shrink the crop's short side below this. With today's crops (~153 px) nothing is shrunk. |

### `ocr.read_validation`

| Key | Default | Meaning |
|---|---|---|
| `font_reader.model` | `''` | `''` = `backend/detection/ocr/models/font_reader.npz`. Verification models live in `models/verification/` (see below). |
| `font_reader.min_literal_score` | `-1.0` | `B`, `:` and `/` must look like themselves (mean log-probability). Rejects garbled cuts and wrong orientation as `format_mismatch`. |
| `font_reader.min_digit_confidence` | `0.0` | Lowest allowed digit confidence, otherwise `low_confidence` |
| `template` | `["#### ##:##", "BB ##/##"]` | One entry per printed line: `#` digit, `A` letter, `*` any character, space = field gap, anything else literal. The font reader needs exactly two lines. |
| `value_pattern` | hour/minute/month regex | Regex the canonical code must fully match, otherwise `invalid_value`. `''` disables it. |
| `min_confidence`, `max_corrections`, `max_ignored_tokens` | `0.80`, `1`, `3` | Template-matching settings of `code_validator.py`. The font reader's lines are checked at confidence 1.0, so `min_confidence` never rejects them. |

If the font reader can't be loaded, or doesn't support the template, the app still starts and logs an error. Every read is then NOK with `failure_reason: ocr_engine_error`.

**Verification models**: `backend/detection/ocr/models/verification/font_reader_without_<bottle>.npz` were trained without that bottle (4113, 3030, 5203, 5307). Point `font_reader.model` at one and put that bottle under the camera to check that the reader reads rather than recalls. Set it back to `''` afterwards.

### `ocr.dynamic_roi` – text locator

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Off: the whole frame is read (a warning is logged on every read) |
| `search_roi` | full frame | Fractional region to search in |
| `blackhat_kernel_size`, `blackhat_threshold` | `25`, `32` | Ink detection |
| `reject_long_lines`, `long_line_min_length_px`, `long_line_erase_thickness_px` | `true`, `60`, `6` | Rib-line removal from the ink mask |
| `morph_kernel_w`, `morph_kernel_h` | `9`, `5` | Merge dots into character blobs |
| `min_component_area`, `max_component_area`, `max_component_dim` | `3`, `9000`, `70` | Component size filters |
| `min_cluster_components`, `cluster_max_gap_px` | `4`, `30` | Text-block clustering |
| `line_extend_max_perp_px`, `line_extend_max_gap_px` | `18`, `90` | Extend the block along its own line |
| `padding_px` | `25` | Margin around the located block |
| `debug_save` | `false` | Write per-read debug images and `font_reader.txt` |
| `debug_dir` | `data/debug/dynamic_roi` | Where they go (relative to the project root) |

## `camera`

| Key | Meaning |
|---|---|
| `device_name` | Friendly name to open (Windows, via pygrabber). The device is looked up again by name on every reconnect, so a different camera at index 0 is never used. |
| `index` | Numeric index, used only when `device_name` is empty or lookup isn't available |
| `width`, `height` | Requested resolution (the camera may deliver less; the log shows what it got) |
| `flip` | OpenCV flip code applied to every frame |
| `manual_settings.*` | Optional fixed exposure/gain/white balance; see the tuning notes in `default.yaml` and `QC_tools/camera_stability.py` |

## `hmi`, `web`, `output`

| Key | Default | Meaning |
|---|---|---|
| `hmi.enable_video_feed` | `true` | Off: the HMI shows a "Video feed disabled" placeholder |
| `hmi.stream_quality` | `75` (if absent) | JPEG quality of the live stream |
| `web.host`, `web.port` | `0.0.0.0`, `5000` | HMI address |
| `output.result_dir` | `data/results` | Folder for `<date>.jsonl` |

## `trigger` – PLC over TCP

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Start the TCP trigger server |
| `host`, `port` | `0.0.0.0`, `5001` | Listen address |
| `trigger_byte` | `"0x01"` | Expected trigger byte. Any other byte also triggers, with a warning in the log. |
| `response_ok` | `"OK\n"` | Sent only for an OK result without mode `ocread`. An OCRead OK sends the code text. |
| `response_nok` | `"NOK\n"` | Sent for every NOK, camera failure, start failure or timeout |
| `timeout_s` | `15.0` | Wait this long for a result before answering NOK (`vision_timeout`) |
| `max_text_length` | `64` | Longer OK reads are sent as NOK (`text_too_long`). Keep this at or below the PLC receive buffer. |

## Frame source (`camera`)

| Key | Default | Meaning |
|---|---|---|
| `camera.source` | `"live"` | `"live"` = the camera; `"images"` = still images from `images_dir`, stepped with the arrow keys on the HMI (each step runs a read) |
| `camera.images_dir` | `"data/test_images"` | Folder with test images (png, jpg, bmp, tif), relative to the project root |
