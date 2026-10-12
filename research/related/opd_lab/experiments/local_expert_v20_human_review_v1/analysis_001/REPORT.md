# Phase20 人工复核分析 001

这批反馈的主要价值是校准“完全成功”的判定。它没有支持 certified 的额外收益，也没有把原来 all_teacher 的 9/12 确认为人工成功率。原助手标签、训练和评测档案保持原样；本目录是独立人工版本。

## 输入与覆盖

原始输入：`human_rating/phase20-human-primary-v1-answers.json`，SHA256 `f017eee1c0cb2dbdd1811e854bceedc5d88cad2710f9bc7c6adcaddc7f518920`。逐字节副本为 [answers_source.json](answers_source.json)，可复算结果为 [results.json](results.json)，来源核验为 [receipt.json](receipt.json)。

- 全部12组都有整体意见，8组有逐项标签；共48张图，32张有逐项标签，30张三个分类字段齐全。
- 144个分类字段中，94个已填：68个 yes、21个 no、5个 unclear；另外50个未填。
- 界面 `complete_groups=1` 同时要求抄写每张图的文字，因此不表示只看过一组。分类字段完整的组实际有7组。
- 空白保留为 missing，不用“都不错”“都不好”等组级评论补填逐图字段；unclear 不作为已知失败。整体偏好不等于约束满足。
- 单位是6个熟悉提示词的12个新seed、单步介入生成。单人复核，非12个独立提示词；不覆盖完整生成路径、全部32例损伤或训练候选的9个接纳目标。

## 同一批可比较的7组

H01、H04、H05、H07、H08、H09、H12的四张图都填完了三个分类字段。这是按回答覆盖形成的描述性子集，不是新的预注册主评测集，也不能外推到全部12组。

| 模型 | 人工确认完全成功 | 至少一项明确不满足 | 无明确不满足、仍有不确定 | 助手此前同7组完全成功 |
| --- | ---: | ---: | ---: | ---: |
| 初始 joint128 | 4/7 | 2/7 | 1/7 | 4/7 |
| 全部教师 all_teacher | 5/7 | 1/7 | 1/7 | 6/7 |
| 验证筛选 certified | 4/7 | 3/7 | 0/7 | 4/7 |
| 随机筛选 random_matched | 5/7 | 2/7 | 0/7 | 4/7 |

H04/H05/H07/H08四种模型都通过。H01四种都不通过。剩余差异来自H09和H12。每组都很少，不能据此排方法名次；但本次人工反馈没有补出 certified 优于初始或随机的证据。

## 对旧结论影响最大的三组

| 组 | 原助手判断 | 人工反馈 | 现在能支持的结论 |
| --- | --- | --- | --- |
| H02，两只杯子和 FRESH | all_teacher 完全成功，其他不成功 | B=all_teacher 整体最好，但杯数 unclear、完整可见性未填 | 有相对偏好，完全成功未获确认；不能把 unknown 改写成失败 |
| H09，另一seed的 FRESH | all_teacher 完全成功 | A=all_teacher 整体最好，杯数/完整可见性 yes，文字 unclear | 物体部分获得肯定，但完整联合成功未获确认 |
| H12，绿色椅子和 PEACH | all_teacher 成功，random_matched 文字 unclear，初始和 certified 文字 no | A=random_matched、B=all_teacher 均三项 yes 且并列；C=初始文字 unclear，D=certified 文字 no | 成功并非 all_teacher 独有；初始是否明确失败也有分歧 |

上述三组正是原 all_teacher 相对初始新增的三个“完全成功”。其中两组仍有人工不确定，另一组随机方法同样通过。因此，原9/12是助手临时标签的结果，不能继续表述为经过人工确认的稳定优势；也不能直接把它机械扣成“人工7/12”，因为其他组仍有缺失标签。

分析时复看H02-B与H09-A原始512像素图，淡杯子/淡字确实接近白底。仅凭可能辨认出轮廓或借助提示词猜读，不足以支持稳健的完全成功结论。这是本次分析意见，不覆盖用户记录或原冻结标签。

## 偏好与判定分歧

8组明确选了字母：all_teacher 在H01/H02/H09单独被选，certified 在H04单独被选，random_matched 在H07/H08单独被选；H05是random_matched与初始并列，H12是random_matched与all_teacher并列。H03/H10整体不好，H06/H11整体认可，但没有给出排序。原话和身份映射逐组保留在 results.json。

这说明有些局部画面差异能被人注意到；它既不是所有方法完全相同的证据，也不是谁达到了完整目标或具有稳定胜率的证据。

94个可比字段中有11处与助手不同，其中7处涉及不确定与确定之间的变化，4处是 yes/no 直接相反，全部发生在H01的完整可见性。H01双方均判文字和位置不满足，因此这些可见性分歧不改变联合失败结果。原助手曾把物体类别判断与完整可见性联系起来；用户将四张都判为可见。需在下一版说明中明确“物体是否符合类别”和“是否被裁切/遮挡”的边界，不能擅自改写用户的 yes。

## 对下一步的约束

1. 不需要用户立即补做整轮标注。先把本次争议样本作为开发用的判定校准例；在下一次身份隐藏评测前写清边界，保持“看不清→不确定”的执行一致性。不能为了让某一方法获胜而移动成功门槛。
2. certified 未显示额外收益这一判断不变。“拒绝状态上的保持监督可能过强”仍只是待检验机制，不由本次人评证实。原计划的 masked 消融仍可作为小规模开发诊断，但不能用这些有争议的自动/助手标签独自决定推进成功。
3. 若继续消融，保留原全32例与两条路径，报告确定成功、明确违反和不确定；这7组仅用于解释此次反馈，不替代原评测分母。下一次可报告的收益应体现为可辨认的修正，并另查损伤；正式确认需要新的seed/提示词及独立复核。

本次未启动GPU训练。结果的作用是收紧证据解释，而非宣布方法成功或彻底否定局部专家修正的想法。

## 复算与检查

从 opd_lab 运行，输出目录必须不存在，避免覆盖旧分析：

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/analyze_phase20_human_review.py --answers human_rating/phase20-human-primary-v1-answers.json --output experiments/local_expert_v20_human_review_v1/analysis_recheck
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_phase20_human_review.py
```

已核对48张图及身份映射、全部1567项原Phase20档案哈希；输入原件与副本哈希一致。缺失/不确定、共同分母、并列偏好与评论中字母解析的3项针对性测试通过。
