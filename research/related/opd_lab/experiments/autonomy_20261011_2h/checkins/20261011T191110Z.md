# Phase81准备完毕，因排队窗口未提交

本次接收20261011T191110Z已写真实UTC ack；完成检查UTC 2026-10-11T19:18:16.607589+00:00。
Phase79/80仍COMPLETED0:0，分别14352/765实际GPU秒；无需重复审计或改写。timer2851812存活。

## 本次推进

完成真实模型worker和一键prepare/submit入口，冻结Phase81（local_expert_v81）202项输入/代码/配置/权重/参考文件。新增4项CPU检查通过，包括完整worker数值模拟贯通256端点、1280质量值、64次精确noop回放、teacher加载/删除/恢复及原生teacher转换校验；此前32项未重复运行。共36项不同CPU检查，真实SD3/GPU尚未验收。

固定Phase77最终EMA作为强原生学生前后缀，64训练prompt，Flow40最近sigma0.278、rho0.10；noop/full_teacher/bounded_teacher/同幅matched_random。主看bounded−random joint，同时看相对noop/full、全部任务、五质量、家族和新增/损失。全程无optimizer更新，目标机会不等于训练成功。Phase56半幅不成功的历史结论保留，不扫rho/时步。

在加载教师之前完成所有学生前缀和noop；临时GenEval只作64query预测，删除之后每个query必须精确回放原学生预测及noop后缀。首个教师转移须与固定原生Flow函数maxabs<=1e-6。两张既有Phase77 EMA开发图仅用作字节精确部署参考，不增加开发集效果比较；8质量参考和既有task参考身份固定。确认/最终继续封存，agent读图0。

## 四分区与预算

已实际probe四分区：minilab最早10月13日12:24EDT，scu10月12日09:16EDT；sablab无兼容L40S，跨项目承诺5/6；preempt预计10月12日02:03:50EDT。虽ai-gpu21显示空闲，单独test-only指定该节点仍预计02:04:18EDT；没有凭空闲状态假设能立即运行。
75分钟硬时限导致预计完成03:19EDT，晚于既定campaign完整完成窗口03:11:10EDT。提交入口在reserve之前停止，未创建submission/intent、未启动GPU、未改配方时限来挤窗口。test-only输出的1077874–1077877不是实际提交作业，不能加入watch。

请求预算仍20700/48600秒，新增1/最多2作业，余下最多1项≤16200秒，不退款。Phase81拟请求4500秒，尚未预留。没有任何新效果分数，原Phase79 EMA50对49/128未定，RAW反例和五质量保留。

## 下一次续看

21:11UTC（17:11EDT）先查已有submission/intent/watch以及真实排队变化；如果仍无提交且四分区估计能完整完成，运行已冻结入口submit即可，它会重新核对来源、probe、reserve并记录。不是失败重试：此前没有GPU作业。若窗口仍不满足，保持准备状态，不取消其他项目/新开timer/扫参数。
命令：PYTHONPATH=opd_related/opd_lab/python conda_envs/envs/diffusionopd/bin/python opd_related/opd_lab/scripts/prepare_bounded_expert_study.py submit
不要再次运行prepare或改冻结scope。作业若技术失败不自动重试；CPU报告错误另立修复记录。若完成，核对202输入、64保存query/256完整端点/1280主要质量+40参考、实际裁剪比例/等幅、64noop精确回放/teacher合同/权重未变及账务，解释目标机会与学习收益边界。
