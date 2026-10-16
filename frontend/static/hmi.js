let CURRENT_MODE = "production";
let LAST_APPLIED_MODE = null;

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
        const ocreadTextEl = document.getElementById("ocreadText");

        if (result.status === "OK") {
            statusEl.textContent = "OK";
            statusEl.className = "status ok";
        } else {
            statusEl.textContent = "NOK";
            statusEl.className = "status nok";
        }

        const noResultsText = "No words detected";

        detEl.innerHTML = detections.length
            ? detections
                .map(d => {
                    const value = d.text || d.label || "-";
                    const confidence = Number(d.confidence ?? 0);
                    return `${value} (${(confidence * 100).toFixed(1)}%)`;
                })
                .join("<br>")
            : noResultsText;

        if (ocreadTextEl) {
            if (result.status === "OK") {
                ocreadTextEl.textContent = result.text || "-";
            } else if (result.failure_reason === "no_text_detected") {
                ocreadTextEl.textContent = "(no text detected)";
            } else if (["format_mismatch", "low_confidence", "invalid_value", "text_too_long"].includes(result.failure_reason)) {
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

    } catch (e) {
        console.error(e);
    }
}

async function loadStatus() {
    // cache: "no-store" ensures this always hits the server for a fresh
    // maintenance-access check, rather than a browser-cached /status reply.
    const res = await fetch("/status", { credentials: "same-origin", cache: "no-store" });
    const data = await res.json();
    CURRENT_MODE = data.maintenance_mode ? "maintenance" : "production";

    const machineIdLabel = document.getElementById("machineIdLabel");
    if (machineIdLabel && data.machine_id) {
        machineIdLabel.textContent = data.machine_id;
    }
}

const RESULT_POLL_MS = 100;

async function startResultPolling() {
    await updateResult();
    setTimeout(startResultPolling, RESULT_POLL_MS);
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
   MODE HANDLING
========================= */
function applyMode() {
    const banner = document.getElementById("modeBanner");
    const prodBtn = document.getElementById("startProductionBtn");
    const maintBtn = document.getElementById("startMaintenanceBtn");
    const cameraControls = document.getElementById("cameraControls");
    const cycleTimeSection = document.getElementById("cycleTimeSection");
    const rotateBtn = document.getElementById("rotateCameraBtn");
    const resetBtn = document.getElementById("resetBtn");

    if (rotateBtn) {
        rotateBtn.style.display = CURRENT_MODE === "maintenance" ? "inline-block" : "none";
    }

    if (CURRENT_MODE === "maintenance") {
        banner.textContent = "MAINTENANCE MODE";
        banner.className = "mode-overlay maintenance";

        if (resetBtn) resetBtn.disabled = false;
        prodBtn.style.display = "inline-block";
        maintBtn.style.display = "none";

        if (cameraControls) {
            cameraControls.style.pointerEvents = "auto";
            cameraControls.style.opacity = "1";
        }

        if (cycleTimeSection) {
            cycleTimeSection.style.display = "block";
        }
    }

    if (CURRENT_MODE === "production") {
        banner.textContent = "PRODUCTION MODE";
        banner.className = "mode-overlay production";

        if (resetBtn) resetBtn.disabled = true;
        prodBtn.style.display = "none";
        maintBtn.style.display = "inline-block";

        if (cameraControls) {
            cameraControls.style.pointerEvents = "none";
            cameraControls.style.opacity = "0.4";
        }

        if (cycleTimeSection) {
            cycleTimeSection.style.display = "none";
        }
    }

    LAST_APPLIED_MODE = CURRENT_MODE;
}

/* =========================
   BUTTON EVENTS
========================= */
document.getElementById("startProductionBtn").addEventListener("click", async () => {
    const confirmed = await showConfirmDialog({
        title: "Start Production Mode",
        message: "Maintenance functions will be locked until maintenance mode is entered again.",
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
        applyMode();
    } catch (e) {
        console.error("Failed to switch to production mode:", e);
        showToast("Could not reach the server. Please try again.", { type: "error" });
    }
});

document.getElementById("startMaintenanceBtn").addEventListener("click", async () => {
    const loggedIn = await showLoginDialog(async (password) => {
        try {
            const res = await fetch("/maintenance_mode", {
                method: "POST",
                credentials: "same-origin",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ maintenance_mode: true, password })
            });

            if (!res.ok) {
                let message = "Incorrect password. Access denied.";
                try {
                    const errBody = await res.json();
                    if (errBody && errBody.message) message = errBody.message;
                } catch (e) {
                    // response body wasn't JSON; fall back to the generic message above
                }
                return { ok: false, message };
            }

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

document.getElementById("rotateCameraBtn").addEventListener("click", async () => {
    if (CURRENT_MODE !== "maintenance") return;
    await fetch("/camera_rotation", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({})
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

/* =========================
    INIT
 ========================= */
async function init() {
    try {
        // Resolve mode first so all downstream behavior is mode-aware.
        await loadStatus();
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