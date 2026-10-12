"""Config 垫片:复用仓库的 config/public.py,只把 resolution 覆写为 1024。
不修改受 preflight 管的 fork 仓库(§ no-temp-files-in-managed-repo)。"""
import importlib.util
_REAL = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/code/DiffusionOPSD-rpi/config/public.py"
_spec = importlib.util.spec_from_file_location("_real_public_cfg", _REAL)
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)

def get_config(name: str):
    c = _m.get_config(name)
    try:
        with c.unlocked():
            c.resolution = 1024
    except AttributeError:
        c.resolution = 1024
    return c
