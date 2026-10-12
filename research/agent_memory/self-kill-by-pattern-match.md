---
name: self-kill-by-pattern-match
description: pkill -f / ps 模式匹配会命中调用者自己的命令行(含 heredoc 文本),已栽七次;唯一可靠解是按 cmdline 的 token 位置匹配,且直接扫 /proc 不按 comm 过滤
metadata:
  type: feedback
---

同一个坑栽了**七次**(2026-09-08 ~ 10-03):

1. `grep -q '[w]atch_...'` —— 匹配到自己;
2. heredoc 文本出现在父 shell 的 args 里 —— 被自己的模式命中;
3. `pkill -f 'salloc_arm.sh lr4xinner2'` —— 杀掉 shell,heredoc 还没执行 ⇒ "杀了东西但什么都没建起来";
4. `pkill -f 'watch_srun_eval_preempt.sh lr8x '` —— 同上;
5. 加了 `[ "$pid" = "$SELF" ]` 守卫**仍然中招** —— `$SELF` 是脚本自己的 pid,被命中的是父 shell;
6. `pgrep -f` 自匹配两次 ⇒ 假"2 个 watcher"、陈旧 PID;
7. **2026-10-03(exit 144)**:已经"写进文件再执行"了,脚本里也比对了 `$$`,
   **还是杀了父 shell** —— 因为**创建那个文件的 shell**,其 cmdline 里含整段脚本全文,
   而我在脚本里用的是 `case "$cl" in *"...rbltpcfm "*)` 这种**子串包含**。

**Why:** "写进文件再执行"只挡住了*执行*那一步;**创建**文件的那条命令的 cmdline 仍含全文。
`$$`/`$PPID` 守卫挡不住它(它是更上层的祖先)。子串包含对任何含脚本文本的进程都会命中。

**How to apply:**
- 永远**不用** `pkill -f`。
- **按 token 位置匹配,不按子串包含**:要求 cmdline 的**第 2 个 token 以目标脚本名结尾**
  (即真的是 `bash <某路径>/<script> <arg1> ...` 这种位置结构)。父 shell 的 cmdline
  第 2 个 token 是 `-c`,自然不匹配。
- **锚 basename,不锚绝对路径** —— 同一脚本有相对/绝对两种启动形式(漏掉一半)。
- **直接扫 `/proc/[0-9]*`,不要用 `pgrep -x bash` 过滤** —— `/bin/bash script.sh` 起的进程
  `comm` 是脚本名截断(如 `watch_matched_c`)而不是 `bash`,实测 16 个 watcher 只找到 2 个。
- 本项目已有正确实现:`opd_related/codex_research/list_watchers.sh`(`list` / `kill ARM`)。
  **直接用它,不要再现写** —— 见 [[reuse-the-correct-impl-dont-rewrite]]。
