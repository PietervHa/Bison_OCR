let CURRENT_MODE = "production";
let CURRENT_USER = null;
let VISION_MODE = null;
let currentThreshold = null; // mirrors backend value
let LAST_APPLIED_MODE = null;
let LAST_DATASET_COUNTS_FETCH = 0;
const DATASET_COUNTS_POLL_MS = 5000;
let DATASET_CAPTURE_ACTIVE = false;

function updateVisionModeButtons() {
    const ocrBtn = document.getElementById("modeOcrBtn");
    const ocreadBtn = document.getElementById("modeOcreadBtn");
    const objBtn = document.getElementById("modeObjBtn");

    if (ocrBtn) {
        ocrBtn.classList.toggle("active", VISION_MODE === "ocr");
    }

    if (ocreadBtn) {
        ocreadBtn.classList.toggle("active", VISION_MODE === "ocread");
    }

    if (objBtn) {
        objBtn.classList.toggle("active", VISION_MODE === "object_detection");
    }
}

/* =========================
   RESULT POLLING
========================= */
async function updateResult() {
    try {
        const res = await fetch("/result");
        const data = await res.json();
        const result = data.result || {};
        const detections = Array.isArray(result.detections) ? result.detections : [];

        const statusEl = document.getElementById("status");
        const detEl = document.getElementById("detections");
        const detectionsHeadingEl = document.getElementById("detectionsHeading");
        const dynamicLabelEl = document.getElementById("dynamicLabel");
        const ocreadTextSection = document.getElementById("ocreadTextSection");
        const ocreadTextEl = document.getElementById("ocreadText");

        if (result.status === "OK") {
            statusEl.textContent = "OK";
            statusEl.className = "status ok";
        } else {
            statusEl.textContent = "NOK";
            statusEl.className = "status nok";
        }

        const noResultsText = (VISION_MODE === "ocr" || VISION_MODE === "ocread")
            ? "No words detected"
            : "No objects detected";

        if (detectionsHeadingEl) {
            detectionsHeadingEl.textContent = VISION_MODE === "ocread" ? "Recognized Text" : "Detections";
        }

        detEl.innerHTML = detections.length
            ? detections
                .map(d => {
                    const value = (VISION_MODE === "ocr" || VISION_MODE === "ocread")
                        ? (d.text || d.label || "-")
                        : (d.label || d.text || "-");
                    const confidence = Number(d.confidence ?? 0);
                    return `${value} (${(confidence * 100).toFixed(1)}%)`;
                })
                .join("<br>")
            : noResultsText;

        if (dynamicLabelEl) {
            dynamicLabelEl.style.display = VISION_MODE === "ocread" ? "none" : "";
            if (VISION_MODE === "ocr") {
                dynamicLabelEl.textContent = result.searched_word || "-";
            } else if (VISION_MODE !== "ocread") {
                dynamicLabelEl.textContent = result.label || "-";
            }
        }

        if (ocreadTextSection) {
            ocreadTextSection.style.display = VISION_MODE === "ocread" ? "block" : "none";
        }
        if (ocreadTextEl && VISION_MODE === "ocread") {
            if (result.status === "OK") {
                ocreadTextEl.textContent = result.text || "-";
            } else if (result.failure_reason === "no_text_detected") {
                ocreadTextEl.textContent = "(no text detected)";
            } else if (["format_mismatch", "low_confidence", "invalid_value", "no_consensus", "text_too_long"].includes(result.failure_reason)) {
                // Read was made but not accepted (read_validation): show why,
                // plus what was read, so operators can see what it struggled with.
                ocreadTextEl.textContent = `(not accepted: ${result.failure_reason}) ${result.text || ""}`.trim();
            } else if (result.failure_reason) {
                // Technical failure (camera unavailable, OCR engine error,
                // timeout, ...) - kept visually distinct from a genuine
                // "no text found" reject so an operator can tell the two apart.
                ocreadTextEl.textContent = `(technical failure: ${result.failure_reason})`;
            } else {
                ocreadTextEl.textContent = "-";
            }
        }

        const timeValue = result.cycle_time_ms || result.processing_time_ms || 0;
        document.getElementById("time").textContent = timeValue + " ms";

        const counters = data.counters || { ok: 0, nok: 0, total: 0 };
        document.getElementById("okCount").textContent = counters.ok;
        document.getElementById("nokCount").textContent = counters.nok;
        document.getElementById("totalCount").textContent = counters.total;

        // Throttle dataset counts polling to once every DATASET_COUNTS_POLL_MS
        try {
            const now = Date.now();
            if (now - LAST_DATASET_COUNTS_FETCH >= DATASET_COUNTS_POLL_MS) {
                LAST_DATASET_COUNTS_FETCH = now;
                (async () => {
                    try {
                        const r = await fetch('/dataset/counts');
                        if (!r.ok) return;
                        const d = await r.json();
                        const okEl = document.getElementById('liveDatasetOkCount');
                        const defEl = document.getElementById('liveDatasetDefCount');
                        if (okEl && typeof d.ok !== 'undefined') okEl.textContent = d.ok;
                        if (defEl && typeof d.defective !== 'undefined') defEl.textContent = d.defective;
                    } catch (e) {
                        console.error('Failed to fetch dataset counts:', e);
                    }
                })();
            }
        } catch (e) {
            console.error(e);
        }

    } catch (e) {
        console.error(e);
    }
}

async function loadStatus() {
    // cache: "no-store" ensures this always hits the server for a fresh
    // maintenance-access check, rather than a browser-cached /status reply.
    const res = await fetch("/status", { credentials: "same-origin", cache: "no-store" });
    const data = await res.json();
    VISION_MODE = data.vision_mode;
    CURRENT_MODE = data.maintenance_mode ? "maintenance" : "production";
    CURRENT_USER = data.maintenance_mode ? (data.username || null) : null;
    updateVisionModeButtons();

    const machineIdLabel = document.getElementById("machineIdLabel");
    if (machineIdLabel && data.machine_id) {
        machineIdLabel.textContent = data.machine_id;
    }
}

function getPollingIntervalMs() {
    return (VISION_MODE === "ocr" || VISION_MODE === "ocread") ? 100 : 500;
}

async function startResultPolling() {
    await updateResult();
    setTimeout(startResultPolling, getPollingIntervalMs());
}

/* =========================
   THRESHOLD HANDLING
========================= */
async function loadThreshold() {
    try {
        const res = await fetch("/threshold");
        const data = await res.json();

        currentThreshold = data.threshold;
        document.getElementById("thresholdInput").value = currentThreshold;
    } catch (e) {
        console.error("Failed to load threshold:", e);
    }
}

async function loadOcrKeyword() {
    const res = await fetch("/ocr_keyword");
    const data = await res.json();
    document.getElementById("ocrKeywordInput").value = data.ocr_keyword;
}

function updateDatasetCaptureUI() {
    const datasetBox = document.getElementById("datasetCaptureBox");
    const instructions = document.getElementById("datasetCaptureInstructions");
    const toggleBtn = document.getElementById("captureToggleBtn");
    const counterDisplay = document.getElementById("datasetCaptureCounter");

    const visible = VISION_MODE === "object_detection";
    if (datasetBox) {
        datasetBox.style.display = visible ? "block" : "none";
    }

    if (!visible) {
        DATASET_CAPTURE_ACTIVE = false;
    }

    const active = visible && DATASET_CAPTURE_ACTIVE && CURRENT_MODE === "maintenance";

    if (toggleBtn) {
        toggleBtn.textContent = active ? "Stop Capture" : "Start Capture";
        toggleBtn.classList.toggle("capture-active", active);
        toggleBtn.disabled = CURRENT_MODE === "production";
    }

    if (instructions) {
        instructions.hidden = !active;
    }


    if (counterDisplay) {
        counterDisplay.hidden = !active;
    }
}

function startDatasetCapture() {
    if (VISION_MODE !== "object_detection") return;
    DATASET_CAPTURE_ACTIVE = true;
    updateDatasetCaptureUI();
}

function stopDatasetCapture() {
    DATASET_CAPTURE_ACTIVE = false;
    updateDatasetCaptureUI();
}

function toggleDatasetCapture() {
    if (DATASET_CAPTURE_ACTIVE) {
        stopDatasetCapture();
    } else {
        startDatasetCapture();
    }
}


/* =========================
   CAMERA CONNECTION MONITORING
========================= */
let CAMERA_CONNECTED = null;       // null = not polled yet
let CAMERA_EVER_CONNECTED = false; // false the whole time => "missing from the start"
const CAMERA_STATUS_POLL_MS = 1000;

function reloadCameraFeed() {
    const img = document.getElementById("cameraFeed");
    if (!img) return;
    // Cache-bust so the browser opens a brand new MJPEG connection instead
    // of trusting a stream that may have gone stale while the camera was
    // disconnected.
    img.src = "/video_feed?t=" + Date.now();
}

async function pollCameraStatus() {
    try {
        const res = await fetch("/camera_status", { cache: "no-store" });
        const data = await res.json();
        const connected = !!data.connected;

        const overlay = document.getElementById("cameraDisconnectedOverlay");
        const modal = document.getElementById("missingCameraModal");
        const wasConnected = CAMERA_CONNECTED;

        if (connected) {
            // Reload the feed on the transition into "connected" (covers
            // both a reconnect after a drop and the camera showing up for
            // the first time), so the live feed is guaranteed fresh.
            if (wasConnected !== true) {
                reloadCameraFeed();
            }
            CAMERA_CONNECTED = true;
            CAMERA_EVER_CONNECTED = true;
            if (overlay) overlay.hidden = true;
            if (modal) modal.hidden = true;
        } else {
            CAMERA_CONNECTED = false;
            if (!CAMERA_EVER_CONNECTED) {
                // Never seen a frame since the page loaded: camera was
                // missing from the start, so block with the popup.
                if (modal) modal.hidden = false;
                if (overlay) overlay.hidden = true;
            } else {
                // It was working before and just dropped out; recovery is
                // already running in the background, so just show a light
                // "reconnecting" overlay on the feed itself.
                if (overlay) overlay.hidden = false;
                if (modal) modal.hidden = true;
            }
        }
    } catch (e) {
        console.error("Failed to fetch camera status:", e);
    }
}

function startCameraStatusPolling() {
    pollCameraStatus();
    setInterval(pollCameraStatus, CAMERA_STATUS_POLL_MS);
}


/* =========================
   DATASET CAPTURE
========================= */
async function captureDatasetImage(label) {
    try {
        if (!DATASET_CAPTURE_ACTIVE) {
            console.error("Dataset capture is not active.");
            return;
        }

        const res = await fetch('/dataset/capture', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label })
        });

        const data = await res.json();
        if (data && data.success) {
            const okEl = document.getElementById('liveDatasetOkCount');
            const defEl = document.getElementById('liveDatasetDefCount');
            if (okEl && typeof data.ok !== 'undefined') okEl.textContent = data.ok;
            if (defEl && typeof data.defective !== 'undefined') defEl.textContent = data.defective;
        } else {
            console.error('Failed to capture dataset image:', data && data.error ? data.error : data);
        }
    } catch (e) {
        console.error('captureDatasetImage error:', e);
    }
}

document.getElementById("applyThreshold").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;

    const value = parseFloat(document.getElementById("thresholdInput").value);

    if (isNaN(value) || value < 0 || value > 1) {
        showToast("Threshold must be between 0.0 and 1.0", { type: "error" });
        return;
    }

    try {
        const res = await fetch("/threshold", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold: value })
        });

        if (!res.ok) {
            showToast("Failed to apply threshold: not in maintenance mode (session may have expired). Reload the page and try again.", { type: "error" });
            return;
        }

        currentThreshold = value; // freeze this value for production
    } catch (e) {
        console.error("Failed to set threshold:", e);
        showToast("Failed to apply threshold: network error. Check the connection and try again.", { type: "error" });
    }
});

/* =========================
   LOGIN LOG (maintenance mode only)
========================= */
/* escapeHtml() is defined in dialogs.js, loaded before this file */

function formatLoginTimestamp(ts) {
    if (!ts) return "-";
    const d = new Date(ts);
    if (isNaN(d.getTime())) return String(ts);
    return d.toLocaleString();
}

function renderLoginLogRow(entry) {
    const success = !!entry.success;
    const cls = success ? "success" : "fail";
    const statusText = success ? "OK" : "FAILED";
    const user = escapeHtml(entry.username || "unknown");
    const time = escapeHtml(formatLoginTimestamp(entry.timestamp));
    return `<div class="login-log-row ${cls}">` +
        `<span class="login-log-user">${user}</span>` +
        `<span class="login-log-status">${statusText}</span>` +
        `<span class="login-log-time">${time}</span>` +
        `</div>`;
}

async function refreshLoginLog() {
    const listEl = document.getElementById("loginLogList");
    if (!listEl) return;
    try {
        const res = await fetch("/login_log", { credentials: "same-origin", cache: "no-store" });
        if (!res.ok) {
            listEl.textContent = "Unable to load login history.";
            return;
        }
        const data = await res.json();
        const entries = Array.isArray(data.logins) ? data.logins : [];
        listEl.innerHTML = entries.length
            ? entries.map(renderLoginLogRow).join("")
            : "No login attempts recorded.";
    } catch (e) {
        console.error("Failed to load login log:", e);
        listEl.textContent = "Unable to load login history.";
    }
}

document.getElementById("refreshLoginLogBtn")?.addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    refreshLoginLog();
});

document.getElementById("clearLoginLogBtn")?.addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    const confirmed = await showConfirmDialog({
        title: "Clear Login History",
        message: "This also lifts any account lockout currently in effect. This cannot be undone.",
        confirmLabel: "Clear",
        destructive: true
    });
    if (!confirmed) return;

    try {
        const res = await fetch("/login_log", {
            method: "DELETE",
            credentials: "same-origin"
        });
        if (!res.ok) {
            showToast("Could not clear login history.", { type: "error" });
            return;
        }
    } catch (e) {
        console.error("Failed to clear login log:", e);
        showToast("Could not reach the server. Please try again.", { type: "error" });
        return;
    }

    refreshLoginLog();
});

/* =========================
   MODE HANDLING
========================= */
function applyMode() {
    const banner = document.getElementById("modeBanner");
    const input = document.getElementById("thresholdInput");
    const applyBtn = document.getElementById("applyThreshold");
    const ocrApplyBtn = document.getElementById("applyOcrKeyword");
    const prodBtn = document.getElementById("startProductionBtn");
    const maintBtn = document.getElementById("startMaintenanceBtn");
    const visionModeToggle = document.getElementById("visionModeToggle");
    const ocrKeywordSection = document.getElementById("ocrKeywordSection");
    const cycleTimeSection = document.getElementById("cycleTimeSection");
    const rotateBtn = document.getElementById("rotateCameraBtn");
    const resetBtn = document.getElementById("resetBtn");

    if (rotateBtn) {
        rotateBtn.style.display = CURRENT_MODE === "maintenance" ? "inline-block" : "none";
    }

    const loginLogBox = document.getElementById("loginLogBox");

    if (CURRENT_MODE === "maintenance") {
        banner.textContent = CURRENT_USER ? `MAINTENANCE MODE — ${CURRENT_USER}` : "MAINTENANCE MODE";
        banner.className = "mode-overlay maintenance";

        input.disabled = false;
        applyBtn.disabled = false;
        if (ocrApplyBtn) ocrApplyBtn.disabled = false;
        if (resetBtn) resetBtn.disabled = false;
        const captureToggleBtn = document.getElementById("captureToggleBtn");
        if (captureToggleBtn) captureToggleBtn.disabled = false;
        prodBtn.style.display = "inline-block";
        maintBtn.style.display = "none";

        if (visionModeToggle) {
            visionModeToggle.style.pointerEvents = "auto";
            visionModeToggle.style.opacity = "1";
        }

        if (cycleTimeSection) {
            cycleTimeSection.style.display = "block";
        }

        if (loginLogBox) {
            loginLogBox.style.display = "block";
            refreshLoginLog();
        }

        if (currentThreshold !== null) input.value = currentThreshold;
    }

    if (CURRENT_MODE === "production") {
        banner.textContent = "PRODUCTION MODE";
        banner.className = "mode-overlay production";

        input.disabled = true;
        applyBtn.disabled = true;
        if (ocrApplyBtn) ocrApplyBtn.disabled = true;
        if (resetBtn) resetBtn.disabled = true;
        const captureToggleBtn = document.getElementById("captureToggleBtn");
        if (captureToggleBtn) captureToggleBtn.disabled = true;
        prodBtn.style.display = "none";
        maintBtn.style.display = "inline-block";

        if (visionModeToggle) {
            visionModeToggle.style.pointerEvents = "none";
            visionModeToggle.style.opacity = "0.4";
        }

        if (cycleTimeSection) {
            cycleTimeSection.style.display = "none";
        }

        if (loginLogBox) {
            loginLogBox.style.display = "none";
        }

        if (currentThreshold !== null) input.value = currentThreshold;
    }

    LAST_APPLIED_MODE = CURRENT_MODE;

    if (ocrKeywordSection) {
        ocrKeywordSection.style.display = (CURRENT_MODE === "maintenance" && VISION_MODE === "ocr")
            ? "flex"
            : "none";
    }

    updateDatasetCaptureUI();
}

async function applyVisionMode(mode) {
    VISION_MODE = mode;
    updateVisionModeButtons();

    const ocrKeywordSection = document.getElementById("ocrKeywordSection");
    if (ocrKeywordSection) {
        ocrKeywordSection.style.display = (CURRENT_MODE === "maintenance" && VISION_MODE === "ocr")
            ? "flex"
            : "none";
    }

    updateDatasetCaptureUI();

    try {
        await fetch("/vision_mode", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ vision_mode: mode })
        });

        await updateResult();
    } catch (e) {
        console.error("Failed to sync vision mode:", e);
    }
}

/* =========================
   BUTTON EVENTS
========================= */
document.getElementById("startProductionBtn").addEventListener("click", async () => {
    const confirmed = await showConfirmDialog({
        title: "Start Production Mode",
        message: "The confidence threshold will be locked until maintenance mode is entered again.",
        confirmLabel: "Start Production",
        destructive: false
    });
    if (!confirmed) return;

    try {
        const res = await fetch("/maintenance_mode", {
            method: "POST",
            credentials: "same-origin",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ maintenance_mode: false })
        });

        if (!res.ok) {
            showToast("Could not switch to production mode.", { type: "error" });
            return;
        }

        CURRENT_MODE = "production";
        CURRENT_USER = null;
        applyMode();
    } catch (e) {
        console.error("Failed to switch to production mode:", e);
        showToast("Could not reach the server. Please try again.", { type: "error" });
    }
});

document.getElementById("startMaintenanceBtn").addEventListener("click", async () => {
    const loggedIn = await showLoginDialog(async (username, password) => {
        try {
            const res = await fetch("/maintenance_mode", {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ maintenance_mode: true, username, password })
            });

            if (!res.ok) {
                let message = "Incorrect username or password. Access denied.";
                try {
                    const errBody = await res.json();
                    if (errBody && errBody.message) message = errBody.message;
                } catch (e) {
                    // response body wasn't JSON; fall back to the generic message above
                }
                return { ok: false, message };
            }

            const data = await res.json();
            CURRENT_USER = data.username || username;
            return { ok: true };
        } catch (e) {
            console.error("Failed to enter maintenance mode:", e);
            return { ok: false, message: "Could not reach the server. Please try again." };
        }
    });

    if (!loggedIn) return;

    CURRENT_MODE = "maintenance";
    applyMode();
});

document.getElementById("modeOcrBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("ocr");
});

document.getElementById("modeOcreadBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("ocread");
});

document.getElementById("modeObjBtn").addEventListener("click", () => {
    if (CURRENT_MODE !== "maintenance") return;
    applyVisionMode("object_detection");
});

document.getElementById("rotateCameraBtn").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    await fetch("/camera_rotation", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
    });
});

document.getElementById("applyOcrKeyword").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    const value = document.getElementById("ocrKeywordInput").value.trim();
    if (!value) { showToast("Search keyword cannot be empty.", { type: "error" }); return; }
    await fetch("/ocr_keyword", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ocr_keyword: value })
    });
});

document.getElementById("resetBtn").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    const confirmed = await showConfirmDialog({
        title: "Reset Counters",
        message: "This will reset the OK, NOK, and Total counters to zero. This cannot be undone.",
        confirmLabel: "Reset",
        destructive: true
    });
    if (!confirmed) return;
    const res = await fetch("/reset_counters", { method: "POST", credentials: "same-origin" });
    if (!res.ok) {
        showToast("Reset failed: not in maintenance mode (session may have expired). Reload the page and try again.", { type: "error" });
    }
});

document.getElementById("captureToggleBtn").addEventListener("click", () => {
    toggleDatasetCapture();
});


/* Keyboard shortcuts for dataset capture */
document.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();

    if (key === "1" && DATASET_CAPTURE_ACTIVE) {
        e.preventDefault();
        captureDatasetImage("ok");
    } else if (key === "2" && DATASET_CAPTURE_ACTIVE) {
        e.preventDefault();
        captureDatasetImage("defective");
    } else if (key === "3" && DATASET_CAPTURE_ACTIVE) {
        e.preventDefault();
        stopDatasetCapture();
    }
});

/* =========================
    INIT
 ========================= */
async function init() {
    try {
        // Resolve mode first so all downstream behavior is mode-aware.
        await loadStatus();
        await loadThreshold();
        await loadOcrKeyword();
        applyMode();

        startResultPolling();
        startCameraStatusPolling();
    } catch (e) {
        console.error("Initialization failed:", e);
    }
}

window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
        window.location.reload();
    }
});

document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
        loadStatus().then(applyMode);
    }
});

init();