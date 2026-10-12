# Phase73 执行记录

用户2026-10-08明确继续；本轮独立一次质量补评，900 L40S请求秒，不使用旧预算。
已提交1074344，minilab-gpu/minilab/high，ai-gpu13，Requeue0、Restarts0。
四分区probe显示sablab跨项目运行7、pending4，合计11，已排除本次sablab提交，未修改其他项目作业。
其他项目状态可能在检查之间变化；本作业只新增minilab一张卡。

第一次启动提交脚本时使用了冻结framework的PYTHONPATH，导致其中ROOT解析为快照内部路径，
在读取inputs.json前退出；未执行probe/reserve/sbatch。随后使用项目live PYTHONPATH运行原封存
提交脚本，正常定位scope，验证296项输入、fresh probe、原子预留并仅提交一次。
这是CPU调用路径修正，未改变科学协议/冻结实现/预算，不是GPU自动重试。

15项冻结CPU测试通过，328张现存图像已由程序校验SHA；agent未读取或查看图像。
完整科学解释仍须等待1600主质量与40参考质量和真实Slurm完成。
