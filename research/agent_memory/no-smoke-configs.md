---
name: no-smoke-configs
description: 永不使用 smoke/缩减配置跑实验;守卫必须校验"被启动的那个 config",不是另一个
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-08-31T10:05:00.000Z
---

用户 2026-08-29 明确要求:**实验配置永远是正式的 —— 完整数据、完整设置。
smoke 只能用来验证代码正确与否,绝不能用来产出任何测量值或结论。**

**Why(同一类错误犯了两次):**
1. 第一次:0010 用 `codex_cr_smoke()` 跑 200-epoch 并据此判负。该配置三个 task slot
   全指向 tiny `testdata/pickscore`(256 条),OCR/GenEval 约束从未在各自分布上被测量。
2. 第二次(更隐蔽):我新建了正式配置 `codex_cr_prod` 并把**守卫**改成查它,
   却漏改**启动行**(仍写死 `--config config/opd.py:codex_cr_smoke`)。
   守卫全绿、日志打印三个不同数据集,但跑的仍是 smoke。
   波及 0015(CR 生产跑)与 0016(梯度几何探针),两者产出的结论全部作废 ——
   包括"两个约束都被守住"和"cos(g,a)≈0 ⇒ QP 值得做"。

**关键教训:守卫在另一个进程里 `get_config(X)`,而启动的是 `get_config(Y)` ——
守卫再严也只是在验证一个从未被执行的对象。** 判据必须落在被启动的进程内。

**How to apply:**
- slurm 里 config 名**收敛成单一变量**(`OPD_EXPECT_CONFIG`),守卫/echo/启动三处共用;
  字面量重复出现就是这个 bug 的温床。
- 在**被启动的进程内**设闸:`codex_cr_smoke` 现已在函数开头要求
  `OPD_ALLOW_SMOKE=1` 才肯返回,否则直接 raise。
- 结论只认**运行时日志里 trainer 自己 dump 的 teachers 段**,不认守卫的打印。
  核对 `- dataset:` 三行互不相同且不含 `testdata`。
- 数据集要校验**实体**不只是路径字符串:文件存在 + 行数匹配
  (pickscore 25432 / ocr 19653 / geneval 50000;注意 ocr 末尾无换行,`wc -l` 会少数 1)。
- **守卫本身要正反两测**:只测"干净输入 → 通过"等于没测。必须再植入一个应被拦截的
  输入,确认它**真的**被拦截。2026-08-31 实测:`preflight_check.py` 的 fork 差异检查
  用**子串匹配**放行文件名,仓库里一个 `train_opsd_ri_sd3.py.bak_pre_n2` 备份就被当成
  "已声明文件"放行;改成精确路径后,修复代码里 `import re` 的插入锚点又静默 no-op,
  也是靠"反向测试"才暴露。
- 新配置写完后若从未真正启动过,它一定是**未验证**的:
  `codex_cr_prod` 就漏了整个 RTC 开关块(缺 `rtc=True` 会直接 raise),
  且 `save_dir` 用相对路径会落在 mktemp 临时目录里、checkpoint 随作业消失。

相关: [[opd-caopd-project]] [[opd-need-positive-results]]
