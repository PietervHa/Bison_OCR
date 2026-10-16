"""
Stand-ins for paddle_worker._worker_loop, used by test_paddle_worker.py.

They live in their own small module (not in the test file) because the
worker runs under the "spawn" start method: the child process has to import
whatever module the target function lives in, and this keeps that import
cheap. Each fake follows the real loop's protocol: report "__ready__" (or
"__init_error__"), then answer (job_id, frame) requests on response_q.

The real loop's `lang` argument is reused as a scratch value (a marker file
path) by the fakes that need state across process restarts.
"""
import os
import time


def _serve(request_q, response_q, handler):
    response_q.put(("__ready__", None))
    while True:
        job_id, frame = request_q.get()
        if job_id is None:
            return
        handler(job_id, frame)


def loop_ok(request_q, response_q, *args):
    _serve(request_q, response_q,
           lambda job_id, frame: response_q.put((job_id, [{"text": "4113 08:25", "confidence": 0.99}])))


def loop_init_error(request_q, response_q, *args):
    response_q.put(("__init_error__", "Traceback (most recent call last):\nValueError: unknown model 'PP-OCRvX_det'"))


def loop_exit_without_ready(request_q, response_q, *args):
    os._exit(7)


def loop_predict_error(request_q, response_q, *args):
    _serve(request_q, response_q,
           lambda job_id, frame: response_q.put((job_id, {"__error__": "ConvertPirAttribute2RuntimeAttribute not support"})))


def loop_hang(request_q, response_q, *args):
    _serve(request_q, response_q, lambda job_id, frame: time.sleep(600))


def loop_crash_first_job_only(request_q, response_q, marker_path, *args):
    """Crashes on its first job ever (across restarts), then works."""
    def handler(job_id, frame):
        if not os.path.exists(marker_path):
            with open(marker_path, "w") as fh:
                fh.write("crashed")
            os._exit(3)  # simulates a native crash: no response, process gone
        response_q.put((job_id, [{"text": "after restart", "confidence": 0.9}]))
    _serve(request_q, response_q, handler)
