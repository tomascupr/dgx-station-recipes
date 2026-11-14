"""Host checks that must pass before the M3 compile.

dsv41 can be up while MemAvailable still covers the spill tier. These checks
do not look at docker. The shell treats a docker error as its own hard fail.
"""
from __future__ import annotations

import os
import subprocess
import sys

from gpus import parse_apps_csv, parse_gpu_csv, role_rows

NEEDLES = ("VLLM::EngineCore", "peer_server3")
DSV41_PORT = 30003


def cmdline_hits(cmdlines: list[str]) -> list[str]:
    return [line for line in cmdlines if any(needle in line for needle in NEEDLES)]


def port_listening(tcp_text: str, port: int) -> bool:
    hexport = f"{port:04X}"
    for line in tcp_text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] == "sl":
            continue
        if parts[3] != "0A":
            continue
        if parts[1].rsplit(":", 1)[-1].upper() == hexport:
            return True
    return False


def gpu_blockers(gpu_rows: list[dict], app_rows: list[dict], idle_mib: int) -> list[str]:
    problems = []
    if len(role_rows(gpu_rows, "gb300")) != 1 or len(role_rows(gpu_rows, "rtx")) != 1:
        names = [row["name"] for row in gpu_rows]
        problems.append(f"need one GB300 and one RTX PRO 6000, saw {names}")
    for app in app_rows:
        problems.append(f"compute app pid {app['pid']} {app['process']} on {app['uuid']}")
    for gpu in gpu_rows:
        if gpu["used_mib"] >= idle_mib:
            problems.append(
                f"{gpu['name']} {gpu['uuid']} memory.used {gpu['used_mib']} MiB "
                f"is not under {idle_mib} MiB")
    return problems


def _cmdlines() -> list[str]:
    me = os.getpid()
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == me:
            continue
        path = f"/proc/{entry}/cmdline"
        try:
            raw = open(path, "rb").read()
        except OSError:
            continue
        if raw:
            found.append(raw.replace(b"\0", b" ").decode(errors="replace"))
    return found


def _tcp() -> str:
    chunks = []
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            chunks.append(open(name).read())
        except OSError:
            continue
    return "\n".join(chunks)


def main() -> None:
    idle = int(os.environ.get("INKLING2_GPU_IDLE_MIB", "2048"))
    hits = cmdline_hits(_cmdlines())
    if hits:
        print("dsv41 still has a process:", file=sys.stderr)
        for line in hits:
            print(line, file=sys.stderr)
        raise SystemExit(1)
    if port_listening(_tcp(), DSV41_PORT):
        raise SystemExit(f"something is listening on :{DSV41_PORT}")
    gpus = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,memory.used",
         "--format=csv,noheader,nounits"],
        text=True)
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
         "--format=csv,noheader"],
        text=True)
    problems = gpu_blockers(parse_gpu_csv(gpus), parse_apps_csv(apps), idle)
    if problems:
        print("GPUs are not idle. This script will not continue.", file=sys.stderr)
        for line in problems:
            print(line, file=sys.stderr)
        raise SystemExit(1)
    print(f"no EngineCore, no peer_server3, :{DSV41_PORT} closed, "
          f"both GPUs under {idle} MiB with no compute apps")


if __name__ == "__main__":
    main()
