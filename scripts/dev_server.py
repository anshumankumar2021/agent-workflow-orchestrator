"""Serve the demo locally: public/ as static files, /api/agent via the real handler.

    GROQ_API_KEY=... python -m scripts.dev_server   # then open http://localhost:3003
"""
from __future__ import annotations

import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from api.agent import handler as ApiHandler

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")


class Dev(SimpleHTTPRequestHandler):
    _send = ApiHandler._send

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_GET(self):
        if self.path.startswith("/api/agent"):
            return ApiHandler.do_GET(self)
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/agent"):
            return ApiHandler.do_POST(self)
        self.send_error(404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3003"))
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("", port), Dev).serve_forever()
