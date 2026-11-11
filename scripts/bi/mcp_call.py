"""
Minimal MCP client for Superset's official MCP service (streamable HTTP, http://127.0.0.1:5008/mcp).

    python scripts/bi/mcp_call.py list                       # tool names
    python scripts/bi/mcp_call.py schema generate_chart      # one tool's input schema
    python scripts/bi/mcp_call.py call list_datasets '{"request": {...}}'
    python scripts/bi/mcp_call.py call generate_chart @payload.json
"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:5008/mcp"


class Client:
    def __init__(self, url=URL):
        self.url, self.sid, self.n = url, None, 0
        r = self._rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "bsad-bi", "version": "1"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.server = r.get("serverInfo")

    def _send(self, body):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.sid:
            h["Mcp-Session-Id"] = self.sid
        req = urllib.request.Request(self.url, json.dumps(body).encode(), h)
        with urllib.request.urlopen(req, timeout=300) as resp:
            self.sid = resp.headers.get("Mcp-Session-Id") or self.sid
            raw = resp.read().decode("utf-8")
        if not raw.strip():
            return None
        if raw.lstrip().startswith("{"):
            return json.loads(raw)
        msgs = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:")]
        return msgs[-1] if msgs else None

    def _rpc(self, method, params=None):
        self.n += 1
        r = self._send({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}})
        if r is None:
            return {}
        if "error" in r:
            raise RuntimeError(json.dumps(r["error"])[:2000])
        return r["result"]

    def tools(self):
        return self._rpc("tools/list")["tools"]

    def call(self, name, args):
        r = self._rpc("tools/call", {"name": name, "arguments": args})
        out = []
        for c in r.get("content", []):
            out.append(c.get("text", json.dumps(c)))
        if r.get("structuredContent") is not None:
            return r["structuredContent"], r.get("isError", False)
        txt = "\n".join(out)
        try:
            return json.loads(txt), r.get("isError", False)
        except Exception:
            return txt, r.get("isError", False)


if __name__ == "__main__":
    c = Client()
    cmd = sys.argv[1]
    if cmd == "list":
        for t in c.tools():
            print(t["name"], "-", (t.get("description") or "").split("\n")[0][:110])
    elif cmd == "schema":
        t = next(t for t in c.tools() if t["name"] == sys.argv[2])
        print(t.get("description", "")[:3000])
        print(json.dumps(t["inputSchema"], indent=1)[:12000])
    elif cmd == "call":
        a = sys.argv[3] if len(sys.argv) > 3 else "{}"
        args = json.load(open(a[1:])) if a.startswith("@") else json.loads(a)
        res, err = c.call(sys.argv[2], args)
        print(("ERROR " if err else "") + (json.dumps(res, indent=1, default=str) if not isinstance(res, str) else res)[:20000])
