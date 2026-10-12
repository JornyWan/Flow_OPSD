"""§4.847 正确性核验:PeftModel.disable_adapter() 在"default+old"双 adapter 下是否真的给出纯底模输出。
用 qlook1@ck100(old 与底模已明显不同)比三种前向:old adapter / disable_adapter() / 不挂 peft 的纯底模。"""
import sys, torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
L = sys.argv[1]
dev = torch.device("cuda")
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
base = pipe.transformer.to(dev).eval(); base.requires_grad_(False)
g = torch.Generator(device=dev).manual_seed(0)
z = torch.randn(1, 16, 64, 64, generator=g, device=dev)
emb = torch.randn(1, 333, 4096, generator=g, device=dev) * 0.1
pemb = torch.randn(1, 2048, generator=g, device=dev) * 0.1
tt = torch.full([1], 465, device=dev, dtype=torch.long)
def fwd(m):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return m(hidden_states=z, timestep=tt, encoder_hidden_states=emb, pooled_projections=pemb, return_dict=False)[0].float()
v_base = fwd(base)
pm = PeftModel.from_pretrained(base, L, adapter_name="default")
pm.load_adapter(L + "/old", adapter_name="old")
pm.set_adapter("old"); v_old = fwd(pm)
pm.set_adapter("default"); v_def = fwd(pm)
pm.set_adapter("old")
with pm.disable_adapter(): v_dis = fwd(pm)
v_old2 = fwd(pm)   # 退出上下文后 adapter 应恢复
r = lambda a, b: float((a - b).norm() / b.norm())
print("‖v_dis − v_base‖/‖v_base‖ = %.3e   (应 ≈0:disable 后就是底模)" % r(v_dis, v_base))
print("‖v_old − v_base‖/‖v_base‖ = %.3e   (应明显 >0:old 已训练)" % r(v_old, v_base))
print("‖v_def − v_base‖/‖v_base‖ = %.3e" % r(v_def, v_base))
print("‖v_old2 − v_old‖/‖v_old‖ = %.3e   (应 ≈0:退出上下文后 old 恢复)" % r(v_old2, v_old))
