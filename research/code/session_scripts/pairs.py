import sys, os, importlib.util
SP = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
def d(a, b, p, n):
    fa = f"matched_pi_{a}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return "   ⏳         "
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return "%+6.2f(t%+5.1f)" % (r[1] / J.MDE, r[3])
for line in sys.argv[1:]:
    a, b, lab = line.split(":")
    print(f"{lab:42s} f40 " + " / ".join(d(a, b, p, 40) for p in ("drawbench", "unseen1k")) + "   f4 " + " / ".join(d(a, b, p, 4) for p in ("drawbench", "unseen1k")))
