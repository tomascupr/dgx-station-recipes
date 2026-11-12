"""Memory-path microbenchmarks for the cold tier: Grace reading its own memory, Grace matmul, and host->GB300 copies."""
import time, torch
G = 2**30
print("threads", torch.get_num_threads())
x = torch.empty(32 * G, dtype=torch.uint8).random_(0, 255)          # 32 GiB, pageable
for name, f in (("cpu sum (read)", lambda: x.view(-1, 64 * 2**20).sum(1)), ("cpu copy (read+write)", lambda: x.clone())):
    best = 1e9
    for _ in range(3):
        t = time.time(); y = f(); best = min(best, time.time() - t); del y
    print(f"{name}: {32 / best:.0f} GiB/s")
xp = x[: 16 * G].pin_memory(); g = torch.empty(16 * G, dtype=torch.uint8, device="cuda")
for name, src in (("pinned host -> GB300", xp), ("pageable host -> GB300", x[: 16 * G])):
    best = 1e9
    for _ in range(3):
        torch.cuda.synchronize(); t = time.time(); g.copy_(src, non_blocking=False); torch.cuda.synchronize(); best = min(best, time.time() - t)
    print(f"{name}: {16 / best:.1f} GiB/s")
del x, xp, g
# one cold expert on the CPU: 3 matmuls 3584x3072 for a handful of tokens, bf16 vs fp32, all cores
for dt in (torch.float32, torch.bfloat16):
    W = [torch.randn(3072, 3584, dtype=dt), torch.randn(3072, 3584, dtype=dt), torch.randn(3584, 3072, dtype=dt)]
    for n in (1, 16):
        h = torch.randn(n, 3584, dtype=dt); t = time.time(); reps = 50
        for _ in range(reps):
            z = (h @ W[0].T) * (h @ W[1].T); o = z @ W[2].T
        dt_s = (time.time() - t) / reps
        print(f"cpu expert {str(dt)[6:]} x{n} tokens: {dt_s * 1e3:.2f} ms  ({3 * 3072 * 3584 * W[0].element_size() / dt_s / G:.0f} GiB/s of weights)")
# many distinct experts in parallel (the real pattern: each token touches different cold experts)
E = 256; W1 = torch.randn(E, 3072, 3584); h = torch.randn(E, 1, 3584)
t = time.time()
for _ in range(5): o = torch.bmm(h, W1.transpose(1, 2))
d = (time.time() - t) / 5
print(f"cpu bmm over {E} distinct experts (one matrix each, fp32): {d * 1e3:.1f} ms, {W1.numel() * 4 / d / G:.0f} GiB/s")
