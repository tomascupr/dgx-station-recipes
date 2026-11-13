"""Safetensors headers only. Never reads tensor bytes."""
from __future__ import annotations

import json
import struct
import urllib.request

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{name}"


def tensor_nbytes(meta: dict) -> int:
    size = DTYPE_BYTES.get(meta["dtype"])
    if size is None:
        raise ValueError(f"unknown safetensors dtype {meta['dtype']}")
    n = 1
    for d in meta["shape"]:
        n *= int(d)
    return n * size


def parse_header_bytes(buf: bytes) -> dict:
    if len(buf) < 8:
        raise ValueError("safetensors header is shorter than 8 bytes")
    n = struct.unpack_from("<Q", buf)[0]
    if n > 100_000_000:
        raise ValueError(f"safetensors header length {n} is not plausible")
    end = 8 + n
    if len(buf) < end:
        raise ValueError(f"header needs {end} bytes, got {len(buf)}")
    return json.loads(buf[8:end])


def header_from_file(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100_000_000:
            raise ValueError(f"{path}: header length {n} is not plausible")
        return json.loads(f.read(n))


def _read_url(url: str, range_header: str | None, timeout: float) -> bytes:
    headers = {"User-Agent": "inkling2-placement"}
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_header(url: str, timeout: float = 60.0) -> dict:
    """HTTP Range request for the JSON header. Does not download the payload."""
    prefix = _read_url(url, "bytes=0-7", timeout)
    n = struct.unpack("<Q", prefix[:8])[0]
    if n > 100_000_000:
        raise ValueError(f"header length {n} from {url} is not plausible")
    blob = _read_url(url, f"bytes=8-{8 + n - 1}", timeout)
    return json.loads(blob)


def resolve_url(repo: str, rev: str, name: str) -> str:
    return HF_RESOLVE.format(repo=repo, rev=rev, name=name)


def fetch_json(url: str, timeout: float = 60.0):
    return json.loads(_read_url(url, None, timeout))


def list_safetensors(repo: str, rev: str, timeout: float = 60.0) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}"
    info = fetch_json(url, timeout)
    names = [s["rfilename"] for s in info.get("siblings", [])]
    return sorted(n for n in names if n.endswith(".safetensors"))
