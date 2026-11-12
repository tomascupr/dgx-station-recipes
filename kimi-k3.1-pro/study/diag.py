"""Sanity view of a native pass: perplexity by position and by row, top-1 accuracy, and one decoded sample."""
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from kstream import Checkpoint

model_dir, out_dir = sys.argv[1], sys.argv[2]
d = torch.load(f"{out_dir}/final.pt")
final, tokens = d["final"][0], d["tokens"]
N, T, H = final.shape
head = Checkpoint(model_dir).get(["language_model.lm_head.weight"], "cuda")["language_model.lm_head.weight"]
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
nll = torch.empty(N, T - 1); hit = torch.empty(N, T - 1); pred = torch.empty(N, T - 1, dtype=torch.long)
with torch.no_grad():
    for a in range(0, N, 4):
        lp = F.log_softmax((final[a:a + 4, :-1].cuda() @ head.T).float(), -1)
        tgt = tokens[a:a + 4, 1:].cuda()
        nll[a:a + 4] = -lp.gather(2, tgt[..., None])[..., 0].cpu()
        pred[a:a + 4] = lp.argmax(-1).cpu(); hit[a:a + 4] = (lp.argmax(-1) == tgt).float().cpu()
print("ppl all", nll.mean().exp().item(), "top1 acc", hit.mean().item(), "median nll", nll.median().item())
for a, b in ((0, 16), (16, 64), (64, 256), (256, 512), (512, 1023)):
    print(f"pos {a:4d}-{b:4d} ppl {nll[:, a:b].mean().exp().item():8.3f} top1 {hit[:, a:b].mean().item():.3f}")
rp = nll.mean(1).exp()
print("row ppl quantiles", [round(rp.quantile(q).item(), 2) for q in (0, .1, .5, .9, 1)])
print("frac tokens nll>10:", (nll > 10).float().mean().item(), " share of total nll:", (nll[nll > 10].sum() / nll.sum()).item())
for r in (0, int(rp.argmax()), int(rp.argmin())):
    print(f"\n=== row {r} ppl {rp[r]:.2f}")
    print("TEXT:", repr(tok.decode(tokens[r, 200:260].tolist())))
    print("PRED:", repr(tok.decode(pred[r, 199:259].tolist())))
    print("worst:", [(repr(tok.decode(tokens[r, i + 1:i + 2].tolist())), repr(tok.decode(pred[r, i:i + 1].tolist())), round(nll[r, i].item(), 1)) for i in nll[r].topk(8).indices.tolist()])

big = (nll > 10)
from collections import Counter
c = Counter(tokens[:, 1:][big].tolist())
print("\nmost common high-nll targets:", [(repr(tok.decode([t])), n) for t, n in c.most_common(25)])
print("high-nll by position bucket:", [round(big[:, a:b].float().mean().item(), 3) for a, b in ((0, 64), (64, 256), (256, 512), (512, 768), (768, 1023))])
print("rows with >10% high-nll tokens:", (big.float().mean(1) > 0.1).sum().item(), "of", N)
r = int(big.float().mean(1).argmax())
i = int(big[r].float().cumsum(0).searchsorted(5))
print("row", r, "ctx:", repr(tok.decode(tokens[r, max(0, i - 40):i + 30].tolist())))
print("pred:", repr(tok.decode(pred[r, max(0, i - 41):i + 29].tolist())))
