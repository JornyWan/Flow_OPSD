import os, sys
sys.path.insert(0, "."); sys.path.insert(0, "../shim_rpi")
keff = 942
print("  %-8s %6s %8s %10s %12s" % ("nproc", "bs", "accum", "B_eff", "B_eff/k_eff"))
for np_ in (3, 4):
    os.environ["PUBLIC_POLICY_WORLD_SIZE"] = str(np_)
    for m in list(sys.modules):
        if m.startswith("config"): del sys.modules[m]
    from config import public
    c = public.get_config("sd35_open3")
    bs = int(c.sample.train_batch_size); acc = int(c.train.gradient_accumulation_steps)
    B = bs * np_ * acc
    print("  %-8d %6d %8d %10d %12.2f" % (np_, bs, acc, B, B / keff))
