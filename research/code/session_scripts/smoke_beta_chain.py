"""code-correctness smoke:--config.beta=1.0 能否压过 public.py:92 的 0.1。
按 train_opsd_ri_sd3.py:198 的同一机制(config_flags.DEFINE_config_file)复现。"""
import sys
from absl import flags
from ml_collections import config_flags
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
FLAGS = flags.FLAGS
FLAGS(sys.argv)
c = FLAGS.config
print("  beta           = %r" % c.beta)
print("  reward_fn      = %r" % dict(c.reward_fn))
print("  num_epochs     = %r" % c.num_epochs)
print("  opsd.opa_rho   = %r" % c.opsd.opa_rho)
