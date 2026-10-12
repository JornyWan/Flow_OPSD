# Phase38离线记录

直接打开report.html：包括原预设结果与明确标记的事后联合阻断诊断；report_primary.html保留追加诊断前的原数值报告。所有报告无图片，不需localhost或外网。

包内包含协议、CPU预算停止与GPU执行修订、两次尝试的代码/输入指纹和日志、完整区域证据、参考来源、逐图及局部比较统计、测试/校验、源码快照。原PNG、模型权重和运行环境不包含在包内；官方源码commit与权重来源/哈希见candidate.json及运行attestation。参考仍为历史助手暂定标签。

GPU实际0.025小时，助手直接读图0张，无扩散生成和训练。本轮文字开发门槛通过，但局部可用修复没有增加，不采用主评分器；另发现旧审核支持门槛在当前参考与规则下不可达到。

最终completion_manifest.json及package_verification.json留在服务器同目录，不递归放入ZIP自身。重跑脚本依赖opd_lab目录结构及attestation中的固定输入路径，离线包并非环境安装器。
