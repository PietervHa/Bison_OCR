# Project map

```
Bison_OCR/
├── backend/
│   ├── main.py                     # entry point: logging, AppState, camera, web server, TCP server, Q-key loop
│   ├── core/
│   │   ├── camera.py               # capture worker + supervisor, reconnect, device lookup by name
│   │   ├── config_loader.py        # loads config/default.yaml (or --config) and .env -> cfg
│   │   ├── state.py                # AppState: last result, counters, maintenance session, rotation
│   │   ├── tcp_trigger_server.py   # PLC trigger byte in, code text / NOK out
│   │   └── vision.py               # runs OCRead in a worker thread, decides OK/NOK
│   ├── detection/
│   │   ├── ocread/
│   │   │   └── engine.py           # OCRead.read(frame): locate -> crop -> font reader -> validate -> result
│   │   └── ocr/
│   │       ├── text_locator.py     # finds and deskews the printed text block
│   │       ├── dot_text.py         # dot-print cleanup (used by the font reader)
│   │       ├── font_reader.py      # classifier trained on the printer's dot font
│   │       ├── code_validator.py   # print template + value_pattern check
│   │       └── models/
│   │           ├── font_reader.npz # production model
│   │           └── verification/   # models trained without one bottle each
│   ├── output/
│   │   └── result_writer.py        # appends results to data/results/<date>.jsonl
│   └── utils/
│       ├── logger.py               # console + data/logs/vision.log
│       └── roi.py                  # fractional ROI -> pixels (text locator search region)
├── frontend/
│   ├── web.py                      # FastAPI app: HMI routes, video stream, maintenance password
│   ├── templates/hmi.html
│   └── static/                     # hmi.js, dialogs.js, hmi.css, img/
├── QC_tools/
│   ├── q_trigger.py                # PLC trigger test (TCP or Q key)
│   ├── ocread_session_report.py    # report from a results JSONL file
│   └── camera_stability.py         # exposure stability check
├── config/default.yaml
├── tests/                          # unittest suite + fixtures/font_reader/*.png, camera_test.py (manual)
├── docs/                           # this documentation + VisionLinkTest.st (PLC side)
├── data/                           # runtime only, not in git: results/, logs/, debug/
├── .env.example                    # MAINTENANCE_PASSWORD
└── requirements.txt
```

## HTTP routes (`frontend/web.py`)

| Route | Auth | Purpose |
|---|---|---|
| `GET /` | – | HMI page |
| `GET /status` | – | maintenance flag for this browser, machine id |
| `GET /result` | – | last result and counters |
| `GET /video_feed` | – | MJPEG live stream |
| `GET /camera_status` | – | camera connected flag |
| `POST /maintenance_mode` | password | enter/leave maintenance mode |
| `POST /camera_rotation` | maintenance | rotate the image 90° |
| `POST /reset_counters` | maintenance | reset OK/NOK/total |
