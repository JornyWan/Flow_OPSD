import importlib.util, os, sys, torch
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file
R = sys.argv[1]; PRE = sys.argv[2]; DIRS = sys.argv[3]
spec = importlib.util.spec_from_file_location("cross_eval", os.path.join(R, "scripts", "cross_eval.py"))
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
dev = torch.device("cuda")
config = ce.load_config("config/public.py", "sd35_open3")
pipe, tes, tks = ce.build_pipeline(config.pretrained.model, PRE, dev, torch.float16)
tr = pipe.transformer
tr.load_adapter(PRE, adapter_name="work"); tr.set_adapter("default")
th0 = load_file(os.path.join(PRE, "adapter_model.safetensors"), device="cuda")
D = torch.load(DIRS, map_location="cpu", weights_only=False)
keys, rho = D["keys"], float(D["rho50"]); f = list(D["delta"])[0]
n = torch.sqrt(sum((D["delta"][f][k].double() ** 2).sum() for k in keys))
u = {k: (D["delta"][f][k].double() / n).float().to(dev) for k in keys}
k0 = keys[0]
mod_name = k0.replace(".lora_A.weight", "")
pw = dict(tr.named_parameters())
pname = [p for p in pw if p.startswith(mod_name + ".lora_A.work")]
print("param name for work:", pname[:2])
pe, ppe = ce.compute_text_embeddings(["a cat"], tes, tks, 128, dev)
z = torch.randn(1, 16, 64, 64, device=dev, dtype=torch.float16)
def fwd(ad):
    tr.set_adapter(ad)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        o = tr(hidden_states=z, timestep=torch.full([1], 260, device=dev, dtype=torch.long),
               encoder_hidden_states=pe, pooled_projections=ppe, return_dict=False)[0]
    tr.set_adapter("default"); return o.float()
o_def = fwd("default"); o_w0 = fwd("work")
print("work(as loaded) vs default max|diff|:", float((o_w0 - o_def).abs().max()))
outs = {}
for s in (+0.3, -0.3, +0.3):
    st = {k: th0[k] + (s * rho) * u[k] for k in th0}
    res = set_peft_model_state_dict(tr, st, adapter_name="work")
    w = dict(tr.named_parameters())[pname[0]]
    exp = st[k0]
    print(f"s={s:+.1f}: unexpected={len(res.unexpected_keys)} | readback max|w-exp|={float((w - exp).abs().max()):.3e} "
          f"| max|w-th0|={float((w - th0[k0]).abs().max()):.3e}")
    o = fwd("work"); outs[s] = o - o_def
    print(f"   ‖out−def‖={float(outs[s].norm()):.4f}")
print("cos(out(+)−def, out(−)−def) =", float((outs[0.3] * outs[-0.3]).sum() / (outs[0.3].norm() * outs[-0.3].norm())))
