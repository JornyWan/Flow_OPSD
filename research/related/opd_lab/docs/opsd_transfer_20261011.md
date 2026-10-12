# OPSD 对当前 OPD 学习问题的可迁移部分

2026-10-11用户提出同时参考OPSD。此记录是方法核对与候选建议，不改变冻结实验，不新增GPU授权或提交。

来源：作者[项目页](https://diffusionopsd.github.io/)、[论文摘要](https://arxiv.org/abs/2608.24646)，及项目页链接的worldbench公开实现。本地固定commit为`9c4c48cdfe5c3fa712d32957345d75fae9295696`。仓库README自称external implementation，不能冒称作者原始训练代码。论文HTML本次无法获取，以下实现细节以固定代码为准，未直接读取任何图片。

## 核实的机制

- `scripts/train_opsd_ri_sd3.py:1251`附近：行为策略真实rollout低噪声查询zq，干净latent预测y0=zq−sigma*v_old。不是拿最终图随意重加噪。
- `_opa_tr_step`（同文件265行）：在VAE解码后的可微reward上对clean latent求梯度，归一化后做上升/下降；每步投影到以y0为中心、半径rho*norm(y0)的球内；目标detach。
- `config/opsd_defaults.py`固定SD3.5M默认query sigma约0.278、rho0.10、两次目标梯度迭代。`opa_cert=0`，标准路径不逐一通过完整后缀验证目标收益；上升/下降目标不是已证明终点正/负样本。
- 训练处v_pos=beta*v_theta+(1−beta)*v_old，v_neg=(1+beta)*v_old−beta*v_theta；分别转成clean prediction拟合y+/y−。相对旧预测的反射使负分支对学生的导数反向。不是两个目标都用普通学生MSE，也不是无界负MSE。
- loss使用detach的逐样本平均绝对误差归一化与advantage权重；有限拟合后刷新behavior。低噪声、目标半径、拟合尺度、行为刷新是不同因素，不能一次全部移植再归因。
- 论文摘要及项目页明确区分目标构造、同query有限拟合、完整生成；更好的目标不保证更好的有限更新。不能把其完整训练成绩解释为每个局部目标都拟合兑现。

## 与本项目的关系

Phase79两步端点训练EMA50/128对原生49/128未定；Phase80说明固定S0状态上梯度多数同向，同时存在早晚拟合取舍。两轮都没有引入终端验证筛选。
OPSD提示应同时考虑“目标是否接近学生、是否有任务价值、有限更新能否兑现”，而不只延长反传路径或增加fit。
借用其目标表示/受限修正/拟合方式，目标仍来自外部教师时仍属于外部专家OPD。直接改用reward梯度构造目标则变为OPSD或混合路线，需正面对照原OPSD，不能作为本项目的新贡献。

关键代数：在同一z与sigma，y_theta=z−sigma*v_theta，所以clean-output MSE等于sigma²乘velocity MSE；Euler下一状态MSE等于delta_sigma²乘velocity MSE。仅换表示主要是在改时间权重/尺度，不自动增加新信息或跨sampler稳健性。低噪声更便于评价细节是动机，不代表早期结构控制可以丢掉。

## 建议的有序验证（未冻结）

1. 保留已验收原生OPD的覆盖、数据、LR与EMA；先检验低噪声真实query上，外部专家clean预测方向的受限修正能否产生额外联合任务收益。比较无修正、同半径随机方向、教师方向；固定后缀、共同噪声，全部任务与质量。半径是独立前瞻选择，不能根据dev挑选，也不能把旧Phase56半幅实验称为同一个控制。
2. 只有目标端存在机会，才比较同目标下原有OPD拟合与OPSD式拟合，固定目标半径/状态/优化预算并记录实际梯度尺度，测目标收益→有限更新收益→完整部署。负分支另作消融，不能同时更改目标源与loss后归因。
3. OCR exact-match/离散结构joint没有可直接使用的reward梯度。可以用既有终端verifier选择外部教师候选；若另外引入可微surrogate，必须明确标注、计成本、保留原硬指标。PickScore/HPS提升不能替代joint改善。
4. 需同覆盖同量随机控制与原生OPD。若直接使用可微reward梯度，必须加原生OPSD，并将共同可微reward实验和不可微joint实验区分。

与现有PREFIX_SUPERVISION_DRAFT并列审议：前者检验保留逐步监督，本文候选检验目标构造与拟合。不要同一轮把两者、低噪声采样及正负分支全叠加。当前不更改任何完成档案或自动提交新GPU。
