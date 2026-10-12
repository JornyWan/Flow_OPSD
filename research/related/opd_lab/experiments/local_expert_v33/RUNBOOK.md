# Phase33 执行与恢复入口

工作目录为 opd_related/opd_lab，解释器为 ../../conda_envs/envs/diffusionopd/bin/python；CPU 命令设置 OMP_NUM_THREADS=4、PYTHONPATH=python:scripts。不要重新运行已完成的 dataset/run 准备命令覆盖数据；它们使用独占新目录。

- GPU 尝试：runs/le33_confirmation_001，Slurm 1061318。源代码、数据、检查点散列与协议已冻结。该作业仅评价，不训练。
- 监控：runs/le33_monitor_001/latest.json、events.jsonl；最终费用保存在本目录 slurm_status.json。文件系统的新文件可见性可能晚于远端日志数十秒；以最终作业状态、结果文件及内容核验共同判断，不仅依据 RUNNING 标志。
- 完成生成后运行 scripts/review_fresh_confirmation.py prepare，核验600个PNG和6000个Euler转移，生成 runs/le33_review_001 的匿名 items.json / pending.json。prepare 完成前不要看新图；prepare 后也不读 key.json 或按方法汇总结果。
- 原生512逐图查看 pending.json 的每一项。用 scripts/record_fresh_observations.py 从标准输入追加完整案例；每个 A–E 项为 [文字布尔或null,任务布尔或null,完整可见布尔或null,实际文字,说明]。所有120例完成后运行 review_fresh_confirmation.py freeze，之后才可 analyze。方法揭示后不反向修改该版标签。
- scripts/build_fresh_human_review.py 只读取匿名图片与预定子集，不需要方法键或助手标签，可在 prepare 后运行。产物 experiments/local_expert_v33_human_review_v1/review.html 内嵌120张图。独立人类答案不属于助手评分；缺失答案不阻碍记录本轮暂定结果，但阻碍声称独立确认。
- 页面检查使用 FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node scripts/check_fresh_pages.cjs human；report 参数检查最终对比页。浏览器测试导出的 synthetic-browser-test-answers.json 仅是合成测试记录，不得当作人类答案。
- 完成分析后运行 report_fresh_confirmation.py、plot_fresh_confirmation.py、verify_fresh_confirmation.py，再检查 report 页面。分别查看报告桌面/移动截图、人工页截图和结果图。
- 写入结果相关 STATUS.md、NEXT.md、interpretation.json、visual_inspection.json，然后运行 archive_fresh_confirmation.py。最后核验 completion_manifest.json 所列全部散列，不再修改归档文件。

主要分析代码散列在 analysis_attestation.json（生成前），展示及辅助检查代码在 supporting_attestation.json（生成中、观察前）。如发现真实错误，应保存原文件与失败记录，明确记录修订及发生时机，不能无说明地重写冻结记录。Phase32 及更早归档保持不变。

新数据共60个词，120个新种子；全数通过历史去重，但60个模板与任务元数据均有历史重叠。三类任务各20词、40案例。人工子集在生成前固定为每类4词、两个种子，共24例。无最终保留测试生成，无多步数或多采样器结论。
