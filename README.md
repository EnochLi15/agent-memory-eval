# Independent Agent Memory Evaluator

Node 24.18.0。`npm ci && npm run build && npm test`。只通过 `/add`、`/search`、`/health` 访问被测服务；不依赖 service 源码或数据库。权威静态 JSON Schema、OpenAPI 与 SHA-256 位于 `contracts/`。

`node dist/cli.js prepare` 适配 LoCoMo_refined、MemOps 和赛题输入；`run` 执行灌入、检索、Answer、Judge；`report` 聚合实验产物。模型凭据只从环境读取。`scripts/select-data.py` 生成按背景分组、开发与保留集互斥的固定划分。

LoCoMo refined Judge 原脚本固定拷贝在 `python/upstream/`，桥接器不改判分 prompt。Python 依赖见 `python/requirements.txt`。本地量化 Qwen3-14B 可用独立 `scripts/ollama-judge-server.py` 将官方请求的关闭 thinking 选项映射到 Ollama 原生接口；该组件只属于评测器，不添加服务 API。量化权重、上下文长度及本地转换均须记录，不能冒充正式未量化平台成绩。

默认代理评测使用配置的 Answer/Judge。`--mode competition-reproduction` 仅在平台配置明确确认后启用。缺少官方 500+500 选择器、Answer 配置和 MemOps 二值映射时，公开集复现一律单独标注。Judge 失败算未判定；报告保留完整计划分母。已有运行只能显式 `--resume`，改变关键配置不能接着写入同一结果。

## 数据与运行命令

```sh
python3 scripts/download-data.py
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.lock
node --env-file=../.env dist/cli.js run --data .data/memops-dev.json --run-id dev-example --base-url http://127.0.0.1:8088 --judge-kind rubric
```

下载器固定上游 commit，并逐一验证405个原始文件的 SHA-256。`configs/splits.json` 保留所有选择 qid 和背景分组。开发/保留集包含完整输入对话，不筛除干扰内容。LoCoMo 无时区日期按 UTC 转换，并保留原始 Session time；MemOps 的外层 timestamp 是保持 segment/turn 顺序的合成值，原文日期不变。
