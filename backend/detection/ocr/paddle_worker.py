"""
PaddleOCR inference worker process.

Runs the PaddlePaddle/PaddleOCR native runtime in a dedicated OS process,
fully isolated from PyTorch's native runtime in the main process. This is
what actually eliminates the OpenMP/MKL thread-pool contention that made
the classifier run slower after OCR had been used — the two runtimes no
longer share a process, so there's nothing left to contend over.

The main process only ever sends a preprocessed numpy frame across and gets
back a plain list of {"text", "confidence"} dicts - the parsing from
PaddleOCR's own result object happens inside this worker process, not in
paddle_ocr.py. This matters more under PaddleOCR 3.x than it used to:
.predict() returns OCRResult objects (dict subclasses carrying detection
polygons, model settings, and other inference-internal state alongside the
text/scores), and there's no guarantee everything bundled in there is safe
or cheap to pickle across the multiprocessing.Queue boundary. Reducing to
plain str/float before it ever touches the queue sidesteps that question
entirely and keeps the IPC payload small regardless of what future
PaddleOCR versions decide to stuff into their result objects.
"""
from __future__ import annotations
import logging
import multiprocessing as mp
import queue
import time

log = logging.getLogger(__name__)


def _parse_ocr_result(results) -> list:
    """Flattens a PaddleOCR 3.x .predict() return value into plain
    {"text", "confidence"} dicts.

    .predict() returns a list of OCRResult objects, one per input
    page/frame (always one here, since we only ever pass a single frame).
    OCRResult is a dict subclass (paddlex.inference.common.result.
    base_result.BaseResult) exposing rec_texts (list[str]) and rec_scores
    (list[float]) directly via subscript access - confirmed by reading the
    installed paddleocr/paddlex source directly rather than assumed, since
    getting this wrong fails silently (empty results, no exception) rather
    than loudly.
    """
    candidates = []
    if not results:
        return candidates
    for page in results:
        try:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
        except (KeyError, TypeError):
            continue
        # Token boxes, used to put tokens in reading order (paddle_ocr
        # _order_candidates). Optional: missing/odd shapes just mean no box.
        try:
            polys = page["rec_polys"]
        except (KeyError, TypeError):
            polys = None
        for idx, (text, score) in enumerate(zip(texts, scores)):
            if not text or not str(text).strip():
                continue
            cand = {"text": str(text), "confidence": round(float(score), 3)}
            try:
                box = [[float(pt[0]), float(pt[1])] for pt in polys[idx]]
                if len(box) >= 2:
                    cand["box"] = box
            except Exception:
                pass
            candidates.append(cand)
    return candidates


def _worker_loop(
    request_q: mp.Queue,
    response_q: mp.Queue,
    lang: str,
    cpu_threads: int,
    use_angle_cls: bool,
    enable_mkldnn: bool,
    text_detection_model_name: str | None,
    text_recognition_model_name: str | None,
) -> None:
    import os
    # PaddleOCR 3.x moved MKL-DNN control to the enable_mkldnn constructor
    # argument (set explicitly below) rather than reading the old
    # FLAGS_use_mkldnn env var this app relied on under 2.x. The env var is
    # left set too (harmless either way), but enable_mkldnn below is what
    # actually controls it now - see _common_args.py / prepare_common_init_args
    # in the installed paddleocr package if this ever needs re-checking
    # against a future version.
    #
    # This defaults True (real CPU acceleration) rather than False. An
    # earlier version of this migration explicitly set it False, carrying
    # over the *intent* of the old FLAGS_use_mkldnn=0 line - but that env
    # var most likely never did anything under 2.x either (2.x controlled
    # this via utility.py's own enable_mkldnn arg, not that FLAGS_ name),
    # so "preserving" it under 3.x's mechanism - which does work - actually
    # removed real acceleration that was implicitly on the whole time. That
    # cost multiple seconds per detected text region on PP-OCRv6_medium
    # with no acceleration; see ocr.enable_mkldnn in default.yaml if this
    # ever needs turning back off for a specific troubleshooting reason.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("PADDLE_DISABLE_FAST_MATH", "1")
    # Everything that can fail while bringing PaddleOCR up (import errors,
    # a model name that doesn't exist, a model download that fails, a
    # paddlepaddle/paddleocr version mismatch...) is reported back to the
    # main process as "__init_error__" with the full traceback, instead of
    # the process just dying and the main process eventually timing out
    # with a bare queue.Empty that says nothing about what went wrong.
    try:
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)

        import numpy as np
        from paddleocr import PaddleOCR as _PaddleOCR

        # use_textline_orientation (called use_angle_cls in 2.x, still accepted
        # as a deprecated alias but renamed here to avoid the deprecation
        # warning firing on every worker start) loads and runs PaddleOCR's
        # text-direction classifier, so characters that are rotated/
        # upside-down relative to the frame (common with dot-print/
        # pin-stamped markings) get rotated upright before the recognition
        # model sees them, instead of being fed in sideways.
        #
        # use_doc_orientation_classify / use_doc_unwarping are both off - both
        # exist for scanned document pages (whole-page rotation / dewarping),
        # not relevant to an already-cropped product-cap image, and turning
        # them off avoids loading two extra models and running two extra
        # inference passes every single cycle for nothing.
        #
        # text_detection_model_name / text_recognition_model_name pin a
        # specific model tier (e.g. PP-OCRv6_small_det/_rec) instead of
        # accepting whichever tier PaddleOCR defaults to for `lang` - see
        # ocr.text_detection_model_name / ocr.text_recognition_model_name in
        # default.yaml. Passing lang alongside explicit model names triggers a
        # (harmless) UserWarning that lang is ignored - that's expected once a
        # model name is pinned.
        ocr_kwargs = dict(
            lang=lang,
            cpu_threads=cpu_threads,
            use_textline_orientation=use_angle_cls,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=enable_mkldnn,
        )
        if text_detection_model_name:
            ocr_kwargs["text_detection_model_name"] = text_detection_model_name
        if text_recognition_model_name:
            ocr_kwargs["text_recognition_model_name"] = text_recognition_model_name

        ocr = _PaddleOCR(**ocr_kwargs)

        # Pay Paddle's own cold-start cost here, once, at worker startup.
        try:
            ocr.predict(np.zeros((64, 64, 3), dtype=np.uint8))
        except Exception:
            pass
    except BaseException:
        import traceback
        response_q.put(("__init_error__", traceback.format_exc()))
        return

    response_q.put(("__ready__", None))

    while True:
        try:
            job_id, frame = request_q.get()
        except (EOFError, OSError):
            break
        if job_id is None:  # shutdown sentinel
            break
        try:
            result = ocr.predict(frame)
            candidates = _parse_ocr_result(result)
            response_q.put((job_id, candidates))
        except Exception as exc:
            response_q.put((job_id, {"__error__": str(exc)}))


class PaddleWorkerError(RuntimeError):
    """The PaddleOCR worker process could not produce a result: it failed
    to start, died, hung past the inference timeout, or raised inside
    .predict(). Callers must treat this as a technical failure of the read
    cycle, NOT as "no text found" - see paddle_ocr.py _recognize()."""


class PaddleOCRWorker:
    """Main-process handle to the PaddleOCR subprocess.

    .ocr(frame) returns a plain list of {"text", "confidence"} dicts,
    already parsed inside the worker process (see _parse_ocr_result) -
    paddle_ocr.py consumes this directly and doesn't need to know which
    PaddleOCR version, or which raw result-object shape, produced it.

    Failure handling: any failure raises PaddleWorkerError, never returns
    an empty list, so a broken engine can't masquerade as "no text on the
    bottle". The subprocess is supervised:

      - before every call, a dead process (native crash, killed, ...) is
        detected and restarted;
      - a call that exceeds `timeout` terminates the process, since a
        worker that blew its timeout is either hung inside native code or
        still busy with a job nobody is waiting for anymore - either way
        the next call is better served by a fresh process than by queueing
        behind it;
      - restart attempts are rate-limited by restart_cooldown_s, so a
        worker that can't start at all (bad model name, broken install)
        fails each cycle immediately instead of blocking every cycle for
        the full ready timeout.

    Not thread-safe: calls are expected from the single vision worker
    thread (plus warmup() at startup, before that thread exists).
    """

    def __init__(
        self,
        lang: str = "en",
        cpu_threads: int = 4,
        use_angle_cls: bool = True,
        enable_mkldnn: bool = True,
        text_detection_model_name: str | None = None,
        text_recognition_model_name: str | None = None,
        ready_timeout: float = 60.0,
        inference_timeout: float = 10.0,
        restart_cooldown_s: float = 10.0,
    ):
        self._worker_args = (
            lang,
            cpu_threads,
            use_angle_cls,
            enable_mkldnn,
            text_detection_model_name,
            text_recognition_model_name,
        )
        self._ready_timeout = float(ready_timeout)
        self._inference_timeout = float(inference_timeout)
        self._restart_cooldown_s = float(restart_cooldown_s)
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._request_q = None
        self._response_q = None
        self._job_counter = 0
        self._last_start_attempt = float("-inf")
        self._last_start_error: str | None = None

        # Startup failure stays fatal (same as before), but now with the
        # worker's own error message instead of a bare queue.Empty.
        self._start()

    # -- process lifecycle --------------------------------------------------

    def _start(self) -> None:
        """Spawn a fresh worker process with fresh queues and wait for it to
        report ready. Raises PaddleWorkerError on failure."""
        self._last_start_attempt = time.monotonic()
        self._kill()

        # Always new queues: terminating a process that was using a queue
        # can leave that queue's internal pipe/lock in a broken state.
        self._request_q = self._ctx.Queue()
        self._response_q = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_worker_loop,
            args=(self._request_q, self._response_q, *self._worker_args),
            daemon=True,
        )
        self._process.start()

        try:
            tag, payload = self._wait_for_startup_message()
        except queue.Empty:
            alive = self._process.is_alive()
            self._kill()
            if alive:
                self._last_start_error = (
                    f"worker did not report ready within {self._ready_timeout:.0f}s "
                    "(still starting - first run downloading models?)"
                )
            else:
                self._last_start_error = (
                    f"worker process exited during startup with code {self._process_exitcode()} "
                    "(see the console for its output)"
                )
            raise PaddleWorkerError(f"PaddleOCR {self._last_start_error}") from None

        if tag == "__init_error__":
            self._kill()
            self._last_start_error = f"worker failed to initialise:\n{payload}"
            raise PaddleWorkerError(f"PaddleOCR {self._last_start_error}")
        if tag != "__ready__":
            self._kill()
            self._last_start_error = f"unexpected startup message {tag!r}"
            raise PaddleWorkerError(f"PaddleOCR {self._last_start_error}")

        self._last_start_error = None

    def _wait_for_startup_message(self):
        """Wait for "__ready__"/"__init_error__", but stop early if the
        process dies without sending either (e.g. a native crash while
        importing paddle) instead of sitting out the whole ready timeout.
        Raises queue.Empty on timeout or early death."""
        deadline = time.monotonic() + self._ready_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                return self._response_q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    # It may have sent its last message just before exiting.
                    return self._response_q.get(timeout=0.5)

    def _process_exitcode(self):
        try:
            return self._process.exitcode if self._process is not None else None
        except Exception:
            return None

    def _kill(self) -> None:
        """Terminate the current worker process (if any) without waiting
        on its queues."""
        proc = self._process
        if proc is None:
            return
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()
                    proc.join(timeout=5)
        except Exception as exc:
            log.warning("Error while terminating PaddleOCR worker: %s", exc)
        for q in (self._request_q, self._response_q):
            try:
                if q is not None:
                    q.cancel_join_thread()
                    q.close()
            except Exception:
                pass

    def _ensure_running(self) -> None:
        """Restart the worker if its process is gone, respecting the
        restart cooldown. Raises PaddleWorkerError if it isn't running and
        can't be (re)started right now."""
        if self._process is not None and self._process.is_alive():
            return

        if self._process is not None:
            log.error(
                "PaddleOCR worker process is not running (exit code %s); restarting it.",
                self._process_exitcode(),
            )
            self._kill()
            self._process = None

        since_last = time.monotonic() - self._last_start_attempt
        if since_last < self._restart_cooldown_s:
            raise PaddleWorkerError(
                "PaddleOCR worker is down; next restart attempt in "
                f"{self._restart_cooldown_s - since_last:.1f}s"
                + (f" (last error: {self._last_start_error.splitlines()[0]})" if self._last_start_error else "")
            )

        try:
            self._start()
        except PaddleWorkerError as exc:
            log.error("PaddleOCR worker restart failed: %s", exc)
            self._process = None
            raise
        log.warning("PaddleOCR worker restarted successfully.")

    # -- public API -----------------------------------------------------------

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def ocr(self, frame, timeout: float | None = None):
        timeout = self._inference_timeout if timeout is None else float(timeout)
        self._ensure_running()

        self._job_counter += 1
        job_id = self._job_counter
        try:
            self._request_q.put((job_id, frame))
        except Exception as exc:
            self._kill()
            self._process = None
            raise PaddleWorkerError(f"could not send frame to PaddleOCR worker: {exc}") from exc

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                # Poll in short slices so a worker that crashes mid-call is
                # noticed right away instead of after the full timeout.
                resp_id, payload = self._response_q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                alive = self._process is not None and self._process.is_alive()
                if alive and deadline - time.monotonic() > 0:
                    continue  # still working on it
                exitcode = self._process_exitcode()
                # Either way this process is no longer trustworthy: dead,
                # or alive but stuck/too slow with a job nobody will read.
                self._kill()
                self._process = None
                if alive:
                    raise PaddleWorkerError(
                        f"PaddleOCR worker did not answer within {timeout:.1f}s; "
                        "worker terminated, it will be restarted on the next cycle"
                    ) from None
                raise PaddleWorkerError(
                    f"PaddleOCR worker process died during inference (exit code {exitcode}); "
                    "it will be restarted on the next cycle"
                ) from None
            except (EOFError, OSError) as exc:
                self._kill()
                self._process = None
                raise PaddleWorkerError(f"PaddleOCR worker connection lost: {exc}") from exc

            if resp_id != job_id:
                continue  # stale response from a previous call
            if isinstance(payload, dict) and "__error__" in payload:
                raise PaddleWorkerError(f"PaddleOCR worker error: {payload['__error__']}")
            return payload

    def shutdown(self) -> None:
        try:
            if self._request_q is not None:
                self._request_q.put((None, None))
        except Exception:
            pass
        if self._process is not None:
            self._process.join(timeout=5)
        self._kill()
