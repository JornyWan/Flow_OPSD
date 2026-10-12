#!/usr/bin/env python
"""Diagnostic 2 —— teacher correction usefulness (推理时 alpha 探针)。

问题:完整的 teacher target 是否**局部过量 / 超出 student 当前可实现范围**?

做法(纯测量,不训练):
  1. student 完整 rollout 得到轨迹 x_0..x_N
  2. 在步 j 上算 mu_S = student 转移均值, mu_T = teacher 在**同一 student 状态**上的转移均值
  3. 构造 mu_alpha = mu_S + alpha*(mu_T - mu_S),**只用这一步**走到 mu_alpha
  4. 其余步全部用 **student** 走完 -> x_0
  5. 解码打分 R(alpha),扫 alpha

判读:若 R(alpha) 在 alpha<1 处取最大(甚至 alpha=1 反而更差),说明完整 teacher 步局部过量,
     支持 Realizability-Aware OPD;若 R 随 alpha 单调上升,则 teacher target 并不过量。

⚠️ 与训练里的 config.train.opd_alpha 完全不同:那个是把 target 阻尼后**训练**,
   由于 target 锚在 student 自身,迭代不动点仍是 mu_T —— 只是收敛更慢,测不出可实现性。
   本脚本是**一次性推理探针**,才是提案 Diagnostic 2 的原意。

复用 train_sd3_opd 的 helper,保证与训练时的数学完全一致(不自行重写去噪)。
"""
import os, sys, json, argparse
import torch
from diffusers import StableDiffusion3Pipeline

# ⚠️ 顺序要紧:先把命令行参数存下来,再清空 sys.argv 供 absl(train_sd3_opd 里的 config_flags)导入,
# 最后用存下来的参数喂 argparse。直接在 import 前清 argv 会把自己的参数也清掉。
_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _here)
from scripts.train_sd3_opd import (                      # noqa: E402
    compute_teacher_step_mean, unwrap_model, _resolve_solver,
)
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob  # noqa: E402
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_lora", default="", help="空=裸 base 当 student")
    ap.add_argument("--teacher_lora", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--n_prompts", type=int, default=32)
    ap.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    ap.add_argument("--probe_steps", default="2,5,8", help="在哪些 j 上探测(0=噪声最高)")
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--teacher_guidance", type=float, default=4.5)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(_ARGV)

    dev = "cuda"
    sd35 = os.environ.get("SD35_PATH")
    # ⚠️ dtype:照抄 evaluation.py:166-169 的单卡推理布局,而不是训练的多卡布局。
    # 训练里 autocast = nullcontext(use_lora 时,见 train_sd3_opd.py:908),所以它并**不**靠 autocast
    # 化解 dtype —— 而是让 transformer 与 text_encoder **同 dtype**、VAE 单独 fp32。
    # 之前分别试过"全 fp16"和"transformer fp32 + te fp16",都因两者 dtype 不一致而报错。
    # compute_teacher_step_mean 内部把 latent 转 float() 再喂 transformer(它是为训练环境写的,
    # 那里 transformer 是 fp32)。与其逐处迁就,不如整体统一 fp32 —— 单卡推理慢一些但消除全部 dtype 冲突。
    DT = torch.float32
    pipe = StableDiffusion3Pipeline.from_pretrained(sd35)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.to(dev, dtype=DT)
    for _te in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        getattr(pipe, _te).to(dev, dtype=DT)
    pipe.vae.to(dev, dtype=torch.float32)
    tr = pipe.transformer
    tr.requires_grad_(False); pipe.vae.requires_grad_(False)

    # ⚠️ 必须先把 transformer 包成 PeftModel 才有 load_adapter/set_adapter
    # (裸 SD3Transformer2DModel 只有 add_adapter)。这里照抄 train_sd3_opd.py:792-805 的做法。
    from peft import PeftModel, LoraConfig, get_peft_model
    if a.student_lora:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, a.student_lora)
    else:
        # 无 student LoRA:用一个零初始化的空 default adapter 当"裸 base",
        # 这样 set_adapter("default") 等价于 base(LoRA 权重为 0),且与训练路径一致。
        tm = [n for n, m in pipe.transformer.named_modules()
              if isinstance(m, torch.nn.Linear) and ("attn" in n or "ff" in n)]
        pipe.transformer = get_peft_model(
            pipe.transformer,
            LoraConfig(r=32, lora_alpha=64, init_lora_weights=True, target_modules=tm))
    pipe.transformer.load_adapter(a.teacher_lora, adapter_name="teacher", is_trainable=False)
    pipe.transformer.set_adapter("default")
    tr = pipe.transformer

    prompts = [l.strip() for l in open(a.prompts_file) if l.strip()][: a.n_prompts]
    alphas = [float(x) for x in a.alphas.split(",")]
    probe_js = [int(x) for x in a.probe_steps.split(",")]

    from flow_grpo.pickscore_scorer import PickScoreScorer
    scorer = PickScoreScorer(device=dev, dtype=torch.float32)

    class _Cfg:  # 供 _resolve_solver 使用
        class sample: pass
        class train: pass
    cfg = _Cfg(); cfg.sample.solver = "flow"; cfg.sample.deterministic = True
    cfg.sample.noise_level = 0.0
    solver, determ = "flow", True

    results = {j: {al: [] for al in alphas} for j in probe_js}
    B = 2
    for bi in range(0, len(prompts), B):
        pr = prompts[bi:bi + B]
        with torch.no_grad():
            pe, npe, ppe, nppe = pipe.encode_prompt(pr, pr, pr, do_classifier_free_guidance=True)
            emb = torch.cat([npe, pe]); pemb = torch.cat([nppe, ppe])
            # --- student 完整 rollout(手写,全程 DT 单一 dtype) ---
            # 不用 pipeline_with_logprob:它内部 prepare_latents 按 vae.dtype(fp32) 建初始 latent,
            # 与 fp16 的 transformer 冲突。手写可保证 dtype 一致,且 ODE 步与训练同用 sde_step_with_logprob。
            pipe.scheduler.set_timesteps(a.num_steps, device=dev)
            _ts_all = pipe.scheduler.timesteps
            lat_list = []
            # flow matching 的初始 latent 就是标准高斯(无 init_noise_sigma,那是 DDPM 系列的属性)
            z = torch.randn(len(pr), 16, a.resolution // 8, a.resolution // 8,
                            device=dev, dtype=DT)
            lat_list.append(z.clone())
            for _jj in range(a.num_steps):
                _t = _ts_all[_jj].expand(len(pr)).to(dev)
                _np = pipe.transformer(
                    hidden_states=torch.cat([z] * 2), timestep=torch.cat([_t] * 2),
                    encoder_hidden_states=emb, pooled_projections=pemb, return_dict=False)[0]
                _nu, _nt = _np.chunk(2)
                _v = _nu + a.guidance * (_nt - _nu)
                _, _, z, _ = sde_step_with_logprob(pipe.scheduler, _v.float(), _t, z.float(), noise_level=0.0)
                z = z.to(DT)
                lat_list.append(z.clone())
            lat = torch.stack(lat_list, dim=1).float()       # [B, N+1, C, H, W]
            ts = pipe.scheduler.timesteps.to(dev).unsqueeze(0).expand(len(pr), -1)

            for j in probe_js:
                x_j, x_j1, t_j = lat[:, j], lat[:, j + 1], ts[:, j]
                # student 在该步的均值:ODE 下就是实际走到的 x_{j+1}
                mu_S = x_j1
                tr.set_adapter("teacher")
                mu_T = compute_teacher_step_mean(
                    transformer=tr, pipeline=pipe, latents_at_j=x_j, next_latent_at_j=x_j1,
                    timestep_at_j=t_j, cond_embeds=emb, cond_pooled=pemb,
                    neg_embeds=npe, neg_pooled=nppe, guidance_scale=a.teacher_guidance,
                    noise_level=0.0, solver=solver, deterministic=determ,
                ).float()
                tr.set_adapter("default")   # default = student(或零初始化的空 LoRA = base)

                for al in alphas:
                    mu_a = mu_S + al * (mu_T - mu_S)          # 只在这一步偏离
                    # 从 mu_a 出发,**用 student** 走完剩余步
                    z = mu_a.to(DT)
                    for jj in range(j + 1, a.num_steps):
                        t_jj = ts[:, jj]
                        npred = pipe.transformer(
                            hidden_states=torch.cat([z.to(DT)] * 2), timestep=torch.cat([t_jj] * 2),
                            encoder_hidden_states=emb.to(DT), pooled_projections=pemb.to(DT),
                            return_dict=False)[0]
                        nu, nt = npred.chunk(2)
                        v = nu + a.guidance * (nt - nu)
                        _, _, z, _ = sde_step_with_logprob(
                            pipe.scheduler, v.float(), t_jj, z.float(), noise_level=0.0)
                        z = z.to(DT)
                    im = pipe.vae.decode(
                        (z.float() / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor,
                        return_dict=False)[0].float()
                    im01 = (im / 2 + 0.5).clamp(0, 1)
                    sc = scorer(pr, im01) if not hasattr(scorer, "score_from_images") else \
                         scorer.score_from_images(scorer.encode_text(pr), im01)
                    results[j][al].extend([float(x) for x in sc])
        print(f"  batch {bi//B+1}/{(len(prompts)+B-1)//B} done", flush=True)

    out = {str(j): {str(al): (sum(v)/len(v) if v else None) for al, v in d.items()}
           for j, d in results.items()}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\n=== R(alpha) 均值 (行=probe step j, 列=alpha) ===")
    print(f"{'j':>4}" + "".join(f"{al:>9}" for al in alphas))
    for j in probe_js:
        print(f"{j:>4}" + "".join(
            f"{out[str(j)][str(al)]:>9.4f}" if out[str(j)][str(al)] is not None else f"{'·':>9}"
            for al in alphas))
    print("\n判读: 峰值在 alpha<1 -> 完整 teacher 步局部过量(支持 Realizability-Aware OPD)")
    print("      单调上升        -> teacher target 不过量,该方向证据不足")


if __name__ == "__main__":
    main()
