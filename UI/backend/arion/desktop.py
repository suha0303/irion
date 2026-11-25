"""Run the loopback desktop backend for the lifetime of its launcher pipe."""

import json
import os
import sys
import threading

import uvicorn


class DesktopServer(uvicorn.Server):
    """Announce the loopback port after the HTTP listener starts."""

    def wait_for_launcher(self):
        try:
            os.read(sys.stdin.fileno(), 1)
        finally:
            self.should_exit = True

    async def startup(self, sockets=None):
        await super().startup(sockets)
        if not self.started:
            return
        port = self.servers[0].sockets[0].getsockname()[1]
        os.environ["ARION_PORT"] = str(port)
        print(json.dumps({"port": port}), flush=True)
        threading.Thread(target=self.wait_for_launcher, daemon=True).start()


def main():
    server = DesktopServer(
        uvicorn.Config(
            "arion.main:app",
            host="127.0.0.1",
            port=0,
            loop="asyncio",
            access_log=False,
        )
    )

    server.run()
    return 0 if server.started else 1


if __name__ == "__main__":
    raise SystemExit(main())
