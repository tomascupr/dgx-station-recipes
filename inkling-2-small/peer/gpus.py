"""Pick the GB300 and the RTX PRO 6000 by name and UUID.

nvidia-smi index 0 on this Station is the RTX. Callers match a PCI bus id to a
CUDA ordinal, or they put the UUID in CUDA_VISIBLE_DEVICES / ``--gpus device=``.
"""
from __future__ import annotations

import csv
import subprocess

ROLES = {
    "gb300": "GB300",
    "rtx": "RTX PRO 6000",
}


def norm_pci(bus: str) -> str:
    text = bus.strip().lower()
    domain, rest = text.split(":", 1)
    return f"{int(domain, 16):04x}:{rest}"


def parse_gpu_csv(text: str) -> list[dict]:
    rows = []
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 4 or not fields[0].strip():
            continue
        uuid, pci, name, used = (part.strip() for part in fields[:4])
        rows.append({"uuid": uuid, "pci": pci, "name": name, "used_mib": int(used)})
    return rows


def parse_apps_csv(text: str) -> list[dict]:
    rows = []
    for fields in csv.reader(text.splitlines()):
        if len(fields) < 3 or not fields[0].strip():
            continue
        uuid, pid, process = (part.strip() for part in fields[:3])
        rows.append({"uuid": uuid, "pid": pid, "process": process})
    return rows


def role_rows(rows: list[dict], role: str) -> list[dict]:
    needle = ROLES[role]
    return [row for row in rows if needle in row["name"]]


def query() -> list[dict]:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,memory.used",
         "--format=csv,noheader,nounits"],
        text=True)
    return parse_gpu_csv(text)


def find(role: str) -> dict:
    if role not in ROLES:
        raise KeyError(role)
    matches = role_rows(query(), role)
    if len(matches) != 1:
        names = [row["name"] for row in query()]
        raise SystemExit(
            f"expected one {ROLES[role]}, found {len(matches)} among {names}")
    return matches[0]


def cuda_ordinal(pci: str) -> int:
    """CUDA ordinal whose PCI bus id is ``pci``. Not the nvidia-smi index."""
    import cudart
    want = norm_pci(pci)
    found = []
    for ordinal in range(cudart.device_count()):
        got = norm_pci(cudart.pci_bus_id(ordinal))
        found.append(got)
        if got == want:
            return ordinal
    raise SystemExit(f"no CUDA device at {want}; visible PCI ids: {found}")
