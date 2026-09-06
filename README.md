# Independent Agent Memory Evaluator

Node 24.18.0、Python 3。`npm ci && npm run build && npm test` 执行Node与Python回归测试。只通过 `/add`、`/search`、`/health` 访问被测服务；不依赖 service 源码或数据库。权威静态 JSON Schema、OpenAPI 与 SHA-256 位于 `contracts/`。

`node dist/cli.js prepare` 适配 LoCoMo_refined、MemOps 和赛题输入；`run` 执行灌入、检索、Answer、Judge；`report` 聚合实验产物。模型凭据只从环境读取。`scripts/select-data.py` 生成按背景分组、开发与保留集互斥的固定划分。

LoCoMo refined Judge 原脚本固定拷贝在 `python/upstream/`，桥接器不改判分 prompt。Python 依赖见 `python/requirements.txt`。本地量化 Qwen3-14B 可用独立 `scripts/ollama-judge-server.py` 将官方请求的关闭 thinking 选项映射到 Ollama 原生接口；该组件只属于评测器，不添加服务 API。量化权重、上下文长度及本地转换均须记录，不能冒充正式未量化平台成绩。

默认代理评测使用配置的 Answer/Judge。`--mode competition-reproduction` 仅在平台配置明确确认后启用。缺少官方 500+500 选择器、Answer 配置和 MemOps 二值映射时，公开集复现一律单独标注。Judge 失败算未判定；报告保留完整计划分母。已有运行只能显式 `--resume`，改变关键配置不能接着写入同一结果。

`python/prepare_judge_calibration.py` 接收 `--primary`、`--predictions`、`--upstream` JSONL和新的 `--output` 目录，核对两套Judge使用完全相同的保存答案，选取全部分歧及六类题型中正/负判定一致的分层样本。`blind.jsonl` 不含原判分，原判分保存在独立文件；`reviews-pending.jsonl` 全部为空待复核，不能当作人工标签。它只生成校准材料，不修改原分数、不调用记忆服务、不把一致判定当作真值。

## 数据与运行命令

```sh
python3 scripts/download-data.py
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.lock
node --env-file=../.env dist/cli.js run --data .data/memops-dev.json --run-id dev-example --base-url http://127.0.0.1:8088 --judge-kind rubric
```

下载器固定上游 commit，并逐一验证405个原始文件的 SHA-256。`configs/splits.json` 保留所有选择 qid 和背景分组。开发/保留集包含完整输入对话，不筛除干扰内容。LoCoMo 无时区日期按 UTC 转换，并保留原始 Session time；MemOps 的外层 timestamp 是保持 segment/turn 顺序的合成值，原文日期不变。

## 上游许可与署名

LoCoMo-Refined是Snap Research LoCoMo的修改版本；本仓库保留固定提交`887091190789e8d6760e70b9edd696539923dc4f`的[CC BY-NC 4.0许可](python/upstream/LICENSE.txt)及[原始署名说明](python/upstream/NOTICE)。两份文件与未修改Judge源码的SHA256均记入上游manifest。公开数据在本地仅进行了已记录的HTTP字段适配、说话人/caption保留、时间顺序转换与分组划分；本项目不把这些转换称为正式平台输入。MemOps固定源码及其MIT许可保存在`python/upstream/memops/`。


### Independent human calibration reports

`python/report_human_calibration.py --packet PATH --reviews reviews.jsonl --reviewer-roster reviewers.jsonl --output NEW_DIRECTORY` validates the frozen blind/prior hashes, selection size and submitted human labels. It never calls a model or fills missing labels. Reports show primary/upstream false positives, false negatives and agreement by question type and original agreement/disagreement stratum. The disagreement-heavy packet does not estimate population judge accuracy.

Review rows keep the packet's `qid`, `status` (`pending_independent_review`, `reviewed` or `uncertain`), `reviewer`, `reviewer_kind: "human"`, `correct` (boolean only for reviewed, otherwise null), `reason`, and optional `criteria`. Each criterion uses `criterion` (`reference`, `must_include:N`, `must_not_include:N`, or `harmful_extra:N`, zero-based), `verdict` (`satisfied`, `violated`, `uncertain`), `answer_quote` (exact saved-answer substring or empty for absence), and `reason`. Criteria are optional; provided criteria receive identity, quote and internal-consistency checks, not automatic semantic adjudication.

The optional JSONL reviewer roster has `reviewer`, `reviewer_kind: "human"`, `independent_of_model_judging: true`, and a nonempty `attestation` supplied by that reviewer. Without an attestation, structurally valid submitted labels are counted as unconfirmed and excluded from confusion matrices. The tool records self-attestation, not authenticated human identity. Do not generate roster statements or human labels on a reviewer's behalf.

All pending, missing, uncertain and unconfirmed reviews remain open. Even a complete packet reports `ready_for_calibration_review`, not benchmark completion. Outputs require a new directory and never overwrite the original packet, labels, prior verdicts or historical scores. With the current pending 199-row packet, the correct report contains zero binary human labels, zero comparisons and null agreement rates.
