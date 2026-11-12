"""A minimal OpenAI-compatible HTTP front for kserve.py: /v1/chat/completions (streaming or not), /v1/models, /health.

One request is generated at a time; others wait. Kimi's output channels are mapped as OpenAI clients expect: the
`think` channel to `reasoning_content`, everything else to `content`.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kserve import MAX_CONTEXT, cache_stats, generate

NAME = "Kimi-K3.1-Pro-q2"


class Channels:
    """Split the token stream into (channel, text) pieces. Format: <|open|>NAME<|sep|> text <|close|>NAME<|sep|>."""

    def __init__(self, tok, prompt=()):
        self.tok = tok
        self.OPEN, self.CLOSE, self.SEP = (tok.encode(s)[0] for s in ("<|open|>", "<|close|>", "<|sep|>"))
        self.channel, self.header, self.run, self.sent = "response", None, [], 0
        for t in prompt[-16:]:                         # the prompt may end by opening a channel ("think") itself
            self.push(t)
        if self.channel.startswith("message"):
            self.channel = "response"
        self.run, self.sent = [], 0

    def push(self, t):
        """Returns (channel, new text) or None."""
        if self.header is not None:                    # inside <|open|>NAME<|sep|> or <|close|>NAME<|sep|>
            if t == self.SEP:
                kind, ids = self.header
                self.header = None
                if kind == "open":
                    self.channel = self.tok.decode(ids).strip() or "response"
                self.run, self.sent = [], 0
            else:
                self.header[1].append(t)
            return None
        if t in (self.OPEN, self.CLOSE):
            self.header = ("open" if t == self.OPEN else "close", [])
            return None
        self.run.append(t)
        text = self.tok.decode(self.run)
        if text.endswith("�"):                    # half of a multi-byte character: wait for the rest
            return None
        new, self.sent = text[self.sent:], len(text)
        return (self.channel, new) if new else None


def serve(model, tok, stop, port, host="127.0.0.1"):
    lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": NAME, "object": "model", "max_context": MAX_CONTEXT}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids = tok.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True)
            ids = list(ids["input_ids"] if isinstance(ids, dict) else ids)
            max_new = min(int(req.get("max_tokens") or req.get("max_completion_tokens") or 512), MAX_CONTEXT - len(ids))
            if max_new <= 0:
                return self._json(400, {"error": f"the prompt has {len(ids)} tokens; the limit is {MAX_CONTEXT}"})
            stream = bool(req.get("stream"))
            rid, created = f"chatcmpl-{int(time.time() * 1e3)}", int(time.time())
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

            def chunk(delta, finish=None):
                obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": NAME,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            ch, text, n = Channels(tok, ids), {"reasoning_content": "", "content": ""}, 0
            with lock:
                t0 = time.time()
                h0, m0 = cache_stats(model)
                try:
                    for t in generate(model, ids, max_new, stop, float(req.get("temperature", 1.0)),
                                      float(req.get("top_p", 0.95)), req.get("seed")):
                        n += 1
                        piece = ch.push(t)
                        if piece:
                            key = "reasoning_content" if piece[0] == "think" else "content"
                            text[key] += piece[1]
                            if stream:
                                chunk({key: piece[1]})
                except (BrokenPipeError, ConnectionResetError):
                    return
                dt = time.time() - t0
                h1, m1 = cache_stats(model)
            finish = "length" if n >= max_new else "stop"
            usage = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}
            print(f"{len(ids)} + {n} tokens in {dt:.1f}s ({n / max(dt, 1e-9):.2f} tok/s incl. prompt), expert cache misses "
                  f"{(m1 - m0) / max(1, m1 - m0 + h1 - h0):.1%}", flush=True)
            if stream:
                chunk({}, finish)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                msg = {"role": "assistant", "content": text["content"]}
                if text["reasoning_content"]:
                    msg["reasoning_content"] = text["reasoning_content"]
                self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": NAME,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})

    print(f"serving {NAME} on http://{host}:{port}/v1 (context {MAX_CONTEXT} tokens, one request at a time)", flush=True)
    ThreadingHTTPServer((host, port), H).serve_forever()
