# C: 保留幅度(拟合端,官方代码已实现) —— 最该先跑
sbatch --job-name=opsdwf --account=minilab --partition=minilab-gpu --gres=gpu:h100:3 \
  --time=12:00:00 --mem=300G \
  --export=ALL,REWARD=pickscore,UPDATES=100,OPA_LS=0,OPA_CERT=0,OPSD_WF_MODE=frozen,\
OUTPUT_DIR=$OPD_ROOT/outputs/opsd_wf100_sd35_pickscore slurm/opsd_ls.slurm

# B': 多查询点(已修稀释) —— 主项权重 1.0,额外点 w=0.3
sbatch --job-name=opsdmqw --account=minilab --partition=minilab-gpu --gres=gpu:h100:3 \
  --time=12:00:00 --mem=300G \
  --export=ALL,REWARD=pickscore,UPDATES=100,OPA_LS=0,OPA_CERT=0,OPA_MQ=0.15,OPA_MQ_W=0.3,\
OUTPUT_DIR=$OPD_ROOT/outputs/opsd_mqw100_sd35_pickscore slurm/opsd_ls.slurm

# E: 查询点 sigma* 扫描 —— 官方固定 0.278 且论文无消融
for S in 0.15 0.45; do
sbatch --job-name=opsdsig${S/./} --account=minilab --partition=minilab-gpu --gres=gpu:h100:3 \
  --time=12:00:00 --mem=300G \
  --export=ALL,REWARD=pickscore,UPDATES=100,OPA_LS=0,OPA_CERT=0,\
OUTPUT_DIR=$OPD_ROOT/outputs/opsd_sig${S/./}_sd35_pickscore slurm/opsd_ls.slurm
done   # 注意: sigma 需在 opsd_ls.slurm 里加 --config.opsd.opa_query_sigma=${OPA_SIG}
