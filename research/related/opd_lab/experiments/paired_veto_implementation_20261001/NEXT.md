# 执行待办：尚未提交GPU

共享engine已支持semantic_veto_v1；本轮125项CPU检查通过。完整训练和评价的GPU执行链
仍未验收，不能直接提交。RESEARCH_PROTOCOL.md的研究问题和判据已固定。

1. 新建三臂veto study控制器，复用clean_base_study的guard/evaluate逻辑，但新CLI必须
   支持all/veto/random。旧controller把random来源写死为verified，旧报告也假定严格
   正例和Phase68；不要改旧冻结快照或把veto默认为旧verified。
2. 固定三臂共同时间日程、输入/权重/评分资源、最终RAW、实际LR审计和逐批数量检查。
   random用prepare_matched_random从veto的原始输入克隆，绝不复制训练checkpoint。
3. 第一轮三臂初始候选hash应相同；随后各臂轨迹不同，不能把后续状态当配对。A重放
   必须调用新known-harm逻辑并保持unknown，错误不得转成未知；旧positive规则不改。
4. 新报告绑定全部384新终点与1920新质量，旧base128/640质量按来源复用；完整任务/
   质量、mask差、16轮实际更新/LR/梯度norm、成本/Slurm及停止判据齐全。无需新图片诊断。
5. CPU集成测试验证空批、来源/模式/版本拒绝、缺cohort/错图hash/残缺报告失败、判据
   不因少指标而通过。完成后才冻结独立local_expert_v71执行目录并预留预算，四分区
   probe后提交，最大三作业各10800秒，本窗口32400上限不变。

可复用：python/opdlab/clean_base_study.py（评价）、runtime.prepare_matched_random（已支持
新版）、scripts/report_sparse_update_study.py的bind_value与通用统计，但它的主入口/旧
decision不能用作本轮报告。新的phase编号尚未创建，不伪造GPU已执行。
