"""按 p11_abcd_eval 的调用顺序逐段复现,看 ± 在哪一步开始变得相同。"""
import importlib.util, os, sys, torch
from torch.cuda.amp import autocast as torch_autocast
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file
R = sys.argv[1]; PRE = sys.argv[2]; DIRS = sys.argv[3]
spec = importlib.util.spec_from_file_location("cross_eval", os.path.join(R, "scripts", "cross_eval.py"))
ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from diffusionopsd.diffusers_patch.solver import run_sampling
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
pe, ppe = ce.compute_text_embeddings(["a cat on a sofa"], tes, tks, 128, dev)
neg, negp = ce.compute_text_embeddings([""], tes, tks, 128, dev)
z0 = torch.randn(1, 16, 64, 64, device=dev, dtype=torch.float16)
pname = [p for p, _ in tr.named_parameters() if ".lora_A.work" in p][0]
kname = pname.replace(".work", "")


def set_work(s):
    st = {k: th0[k] + (s * rho) * u[k] for k in th0}
    set_peft_model_state_dict(tr, st, adapter_name="work")


def one(ad):
    tr.set_adapter(ad)
    o = tr(hidden_states=z0, timestep=torch.full([1], 260, device=dev, dtype=torch.long),
           encoder_hidden_states=pe, pooled_projections=ppe, joint_attention_kwargs=None, return_dict=False)[0].float()
    tr.set_adapter("default"); return o


def probe(stage):
    outs, ws = {}, {}
    for s in (+0.3, -0.3):
        set_work(s)
        ws[s] = dict(tr.named_parameters())[pname].detach().clone()
        outs[s] = one("work")
    od = one("default")
    print(f"[{stage}] w±相同={torch.equal(ws[0.3], ws[-0.3])}  w₊−θ0={float((ws[0.3]-th0[kname]).abs().max()):.2e}  "
          f"out₊==out₋:{torch.equal(outs[0.3], outs[-0.3])}  ‖out₊−def‖={float((outs[0.3]-od).norm()):.3f} "
          f"‖out₋−def‖={float((outs[-0.3]-od).norm()):.3f}", flush=True)


with torch_autocast(enabled=True, dtype=torch.float16), torch.no_grad():
    probe("autocast+no_grad")
    g = [torch.Generator(device=dev).manual_seed(0)]
    _, Z, _ = pipeline_with_logprob(pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
        negative_prompt_embeds=neg, negative_pooled_prompt_embeds=negp, num_inference_steps=40, guidance_scale=1.0,
        output_type="pt", height=512, width=512, noise_level=0.7, deterministic=True, solver="flow",
        model_type="sd3", generator=g, decode=False)
    probe("after pipeline_with_logprob")
    sig = pipe.scheduler.sigmas.float()
    def vfn(z, sigma):
        tt = torch.full([z.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
        return tr(hidden_states=z, timestep=tt, encoder_hidden_states=pe, pooled_projections=ppe,
                  joint_attention_kwargs=None, return_dict=False)[0].to(pe.dtype)
    ends = {}
    for s in (+0.3, -0.3):
        set_work(s); tr.set_adapter("work")
        ends[s], _, _ = run_sampling(vfn, Z[35], sig[35:], "flow", True, 0.7, max_steps=5)
        tr.set_adapter("default")
    print("[run_sampling C] z₊==z₋:", torch.equal(ends[0.3], ends[-0.3]),
          "‖z₊−z₋‖=%.4f ‖z₊−pre‖=%.4f" % (float((ends[0.3] - ends[-0.3]).float().norm()), float((ends[0.3] - Z[-1]).float().norm())))
import diffusionopsd.rewards
sc = {nm: diffusionopsd.rewards.multi_score(dev, {nm: 1.0}) for nm in ("pickscore", "clipscore", "hpsv2")}
with torch_autocast(enabled=True, dtype=torch.float16), torch.no_grad():
    probe("after loading scorers")
