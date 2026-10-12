# Phase69失败后的独立CPU审计：任务完整，五质量全部缺失

Slurm1065085 FAILED 1:0，实际1296 L40S秒；因新代码缺少cohort字段，在通用质量token审计停止。
这是实现错误，原18项测试漏掉了生产者与真实质量审计接口；不是算法效果失败。原档案不改，不自动重试GPU。

102状态生成、204原PNG精确重放、510任务评分/图像hash和102局部误差均已核验；新增推理0。
**五项质量0/2550，不能报告完整实验通过、质量保持或据此自动推进训练。**

|训练臂|阶段|joint/51|OCR/51|结构/51|
|---|---|---:|---:|---:|
|verified|behavior|1|24|7|
|verified|teacher|33|45|38|
|verified|immediate_local|1|23|7|
|verified|final_local|1|25|7|
|verified|final_suffix|2|25|8|
|random|behavior|4|22|8|
|random|teacher|19|33|27|
|random|immediate_local|4|23|7|
|random|final_local|2|23|6|
|random|final_suffix|2|23|6|

两臂各51查询、35个训练prompt；状态各自内生，不作跨臂配对推断。最终后缀仍共用历史前缀，不是完整部署。

verified：原teacher的B直接joint修复32个，即时局部/最终局部/最终后缀保留0/0/1。
目标loss即时下降3.08%，最终下降12.77%；最终动作的能量加权教师方向投影仅7.64%。

random：原teacher的B直接joint修复15个，即时局部/最终局部/最终后缀保留0/0/0。
目标loss即时下降2.59%，最终下降11.18%；最终动作的能量加权教师方向投影仅6.54%。

任务证据支持“有可执行的局部候选，但当前拟合未充分兑现”，尚不能保证更多拟合步或较大LR改善完整生成；旧高LR已退化。
停止自动GPU推进，先补独立schema修复与测试、准备有限拟合可行性协议；质量缺口明确保留。
