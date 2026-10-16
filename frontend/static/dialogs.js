/* =====================================================================
   MDE AUTOMATION — Dialog system
   Replaces native alert()/confirm()/prompt() with brand-styled equivalents.
   Generic and reusable — knows nothing about specific endpoints. Callers
   supply an async onSubmit/onConfirm callback and this module handles all
   the UI: focus, keyboard, loading state, inline errors.
===================================================================== */

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML;
}

/* ---------- Toasts: transient, non-blocking notifications ---------- */

let _toastContainer = null;
function _ensureToastContainer() {
    if (_toastContainer) return _toastContainer;
    _toastContainer = document.createElement("div");
    _toastContainer.className = "toast-container";
    document.body.appendChild(_toastContainer);
    return _toastContainer;
}

function showToast(message, opts) {
    opts = opts || {};
    const type = opts.type || "error";
    const duration = opts.duration || (type === "error" ? 6000 : 4000);
    const container = _ensureToastContainer();

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-message">${escapeHtml(message)}</span>
        <button class="toast-close" aria-label="Dismiss">&times;</button>
    `;

    let hideTimer = null;
    function remove() {
        toast.classList.remove("toast-show");
        setTimeout(() => toast.remove(), 200);
    }
    toast.querySelector(".toast-close").addEventListener("click", () => {
        if (hideTimer) clearTimeout(hideTimer);
        remove();
    });

    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("toast-show"));
    hideTimer = setTimeout(remove, duration);
    return toast;
}

/* ---------- Generic modal shell ---------- */

function _buildOverlay() {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    overlay.innerHTML = `<div class="modal-box"></div>`;
    document.body.appendChild(overlay);
    return overlay;
}

/* ---------- Confirm dialog ----------
   Returns a Promise<boolean>. For destructive actions, Cancel is focused
   by default and Enter does NOT confirm — only a deliberate click/Space
   on the (unfocused) Confirm button does, so a stray Enter keypress can
   never trigger something destructive. */
function showConfirmDialog(opts) {
    opts = opts || {};
    const title = opts.title || "Are you sure?";
    const message = opts.message || "";
    const confirmLabel = opts.confirmLabel || "Confirm";
    const cancelLabel = opts.cancelLabel || "Cancel";
    const destructive = !!opts.destructive;

    return new Promise((resolve) => {
        const overlay = _buildOverlay();
        const box = overlay.querySelector(".modal-box");
        box.classList.add(destructive ? "destructive" : "neutral");
        box.innerHTML = `
            <h2>${escapeHtml(title)}</h2>
            <p class="modal-message">${escapeHtml(message)}</p>
            <div class="modal-actions">
                <button class="btn reset modal-cancel">${escapeHtml(cancelLabel)}</button>
                <button class="btn ${destructive ? "maintenance" : "production"} modal-confirm">${escapeHtml(confirmLabel)}</button>
            </div>
        `;

        const cancelBtn = box.querySelector(".modal-cancel");
        const confirmBtn = box.querySelector(".modal-confirm");

        function cleanup(result) {
            document.removeEventListener("keydown", onKey);
            overlay.remove();
            resolve(result);
        }
        function onKey(e) {
            if (e.key === "Escape") { cleanup(false); return; }
            if (e.key === "Enter" && !destructive) { cleanup(true); }
            // Destructive: no global Enter binding. Enter only acts through
            // native focused-button behavior, and Cancel holds focus below.
        }
        document.addEventListener("keydown", onKey);
        cancelBtn.addEventListener("click", () => cleanup(false));
        confirmBtn.addEventListener("click", () => cleanup(true));
        overlay.addEventListener("click", (e) => { if (e.target === overlay) cleanup(false); });

        (destructive ? cancelBtn : confirmBtn).focus();
    });
}

/* ---------- Login dialog ----------
   onSubmit(username, password) must return { ok: bool, message?: string }.
   On failure the dialog stays open and shows the message inline, so a
   wrong password doesn't force re-typing the username. Resolves true on
   successful login, false if the user cancels. */
function showLoginDialog(onSubmit) {
    return new Promise((resolve) => {
        const overlay = _buildOverlay();
        const box = overlay.querySelector(".modal-box");
        box.classList.add("neutral", "login-box");
        box.innerHTML = `
            <h2>Maintenance Login</h2>
            <div class="login-error" hidden></div>
            <div class="login-field">
                <label for="loginUsernameInput">Username</label>
                <input type="text" id="loginUsernameInput" autocomplete="username">
            </div>
            <div class="login-field">
                <label for="loginPasswordInput">Password</label>
                <input type="password" id="loginPasswordInput" autocomplete="current-password">
            </div>
            <div class="modal-actions">
                <button class="btn reset login-cancel">Cancel</button>
                <button class="btn maintenance login-submit">Log In</button>
            </div>
        `;

        const userInput = box.querySelector("#loginUsernameInput");
        const passInput = box.querySelector("#loginPasswordInput");
        const errorEl = box.querySelector(".login-error");
        const submitBtn = box.querySelector(".login-submit");
        let submitting = false;

        function showFieldError(msg) {
            errorEl.textContent = msg;
            errorEl.hidden = false;
        }
        function hideFieldError() {
            errorEl.hidden = true;
        }
        function cleanup(result) {
            document.removeEventListener("keydown", onKey);
            overlay.remove();
            resolve(result);
        }

        async function submit() {
            if (submitting) return;
            const username = userInput.value.trim();
            const password = passInput.value;

            if (!username) { showFieldError("Username cannot be empty."); userInput.focus(); return; }
            if (!password) { showFieldError("Password cannot be empty."); passInput.focus(); return; }

            hideFieldError();
            submitting = true;
            submitBtn.disabled = true;
            submitBtn.textContent = "Logging in...";
            try {
                const result = await onSubmit(username, password);
                if (result && result.ok) {
                    cleanup(true);
                    return;
                }
                showFieldError((result && result.message) || "Login failed.");
                passInput.value = "";
                passInput.focus();
            } finally {
                submitting = false;
                submitBtn.disabled = false;
                submitBtn.textContent = "Log In";
            }
        }

        function onKey(e) {
            if (e.key === "Escape") { cleanup(false); return; }
            if (e.key === "Enter") { e.preventDefault(); submit(); }
        }
        document.addEventListener("keydown", onKey);
        box.querySelector(".login-cancel").addEventListener("click", () => cleanup(false));
        submitBtn.addEventListener("click", submit);
        overlay.addEventListener("click", (e) => { if (e.target === overlay) cleanup(false); });

        userInput.focus();
    });
}
