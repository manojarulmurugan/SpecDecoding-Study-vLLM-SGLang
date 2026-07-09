"""In-process fake of a vLLM OpenAI-compatible server.

Implements just enough surface for end-to-end harness tests without a GPU:
  - GET  /health           -> 200
  - GET  /version          -> {"version": "0.0-fake"}
  - GET  /metrics          -> Prometheus text with spec-decode counters that
                              advance on every completion request
  - POST /v1/completions   -> SSE stream: N text chunks, a usage chunk,
                              then [DONE]

The completion text is canned per server instance so correctness scoring
can be exercised deterministically.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional


class FakeVllmServer:
    def __init__(
        self,
        completion_text: str = " The answer is 6.",
        chunk_delay_s: float = 0.005,
        spec_metrics: bool = True,
        drafts_per_request: int = 10,
        draft_tokens_per_request: int = 50,
        accepted_tokens_per_request: int = 30,
    ):
        self.completion_text = completion_text
        self.chunk_delay_s = chunk_delay_s
        self.spec_metrics = spec_metrics
        self.drafts_per_request = drafts_per_request
        self.draft_tokens_per_request = draft_tokens_per_request
        self.accepted_tokens_per_request = accepted_tokens_per_request

        self.request_count = 0
        self.in_flight = 0
        self.max_in_flight_seen = 0
        # capacity-pressure knobs for K-stress tests: kv usage rises with
        # in-flight requests; preemptions accumulate per request
        self.kv_usage_per_request = 0.1
        self.waiting_reported = 0
        self.preemptions_per_request = 0
        self.seen_payloads: List[dict] = []
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> str:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, b"", "text/plain")
                elif self.path == "/version":
                    self._send(200, json.dumps({"version": "0.0-fake"}).encode(),
                               "application/json")
                elif self.path == "/metrics":
                    self._send(200, server._metrics_text().encode(), "text/plain")
                else:
                    self._send(404, b"", "text/plain")

            def do_POST(self):
                if self.path != "/v1/completions":
                    self._send(404, b"", "text/plain")
                    return
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.request_count += 1
                    server.in_flight += 1
                    server.max_in_flight_seen = max(
                        server.max_in_flight_seen, server.in_flight
                    )
                    server.seen_payloads.append(payload)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    words = server.completion_text.split(" ")
                    chunks = [w if i == 0 else " " + w for i, w in enumerate(words)]
                    for i, text in enumerate(chunks):
                        chunk = {
                            "choices": [{
                                "text": text,
                                "index": 0,
                                "finish_reason": "stop" if i == len(chunks) - 1 else None,
                            }],
                        }
                        self._sse(chunk)
                        time.sleep(server.chunk_delay_s)
                    usage_tokens = len(chunks) * 2  # 2 tokens/chunk: spec-decode-like
                    self._sse({
                        "choices": [],
                        "usage": {"prompt_tokens": 11, "completion_tokens": usage_tokens},
                    })
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    with server._lock:
                        server.in_flight -= 1

            def _sse(self, obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.base_url

    @property
    def base_url(self) -> str:
        assert self._httpd is not None
        return "http://127.0.0.1:%d" % self._httpd.server_address[1]

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    # -- metrics -------------------------------------------------------------

    def _metrics_text(self) -> str:
        lines = [
            "# HELP vllm:request_success_total requests",
            'vllm:request_success_total{model_name="fake"} %d' % self.request_count,
            'vllm:num_requests_running{model_name="fake"} %d' % self.in_flight,
            'vllm:num_requests_waiting{model_name="fake"} %d' % self.waiting_reported,
            'vllm:kv_cache_usage_perc{model_name="fake"} %.3f'
            % min(1.0, self.in_flight * self.kv_usage_per_request),
            'vllm:num_preemptions_total{model_name="fake"} %d'
            % (self.request_count * self.preemptions_per_request),
        ]
        if self.spec_metrics:
            n = self.request_count
            lines += [
                'vllm:spec_decode_num_drafts_total{model_name="fake"} %d'
                % (n * self.drafts_per_request),
                'vllm:spec_decode_num_draft_tokens_total{model_name="fake"} %d'
                % (n * self.draft_tokens_per_request),
                'vllm:spec_decode_num_accepted_tokens_total{model_name="fake"} %d'
                % (n * self.accepted_tokens_per_request),
            ]
        return "\n".join(lines) + "\n"
