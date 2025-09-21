# Contributing / 贡献指南

> English follows Chinese section. / 英文版在后半部分。

## 快速导航

- [行为准则](#行为准则)
- [开发环境准备](#开发环境准备)
- [提交前检查清单](#提交前检查清单)
- [Issue / PR 规范](#issue--pr-规范)
- [分支策略](#分支策略)
- [新增配置项流程](#新增配置项流程)
- [代码风格与质量](#代码风格与质量)
- [测试与诊断脚本](#测试与诊断脚本)
- [常见贡献类型建议](#常见贡献类型建议)
- [发布与版本](#发布与版本)

---

## 行为准则

本项目遵循开源社区友好协作原则：尊重、透明、可追溯。禁止人身攻击、歧视、传播违法违规内容。

## 开发环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn graphrag.service.server:app --reload --port 8000
```

可选检查：

```bash
python scripts/check_env_sync.py
python scripts/check_requirements_imports.py
python -m graphrag.diagnostics.dump
```

## 提交前检查清单

- [ ] 新增或修改的环境变量已：`settings.py` 定义 / `.env` 示例 / README.* 与 Features 文档同步
- [ ] 重要逻辑增加最小单测或可复用脚本验证
- [ ] 诊断字段（若有新增）已在 `/diagnostics` 输出并文档化
- [ ] 中英文文档同步（若适用）
- [ ] 无调试遗留打印 / 减少冗余注释 / 代码符合现有风格
- [ ] 通过 lint / import 检查脚本

## Issue / PR 规范

**Issue 模板建议包含**：背景、预期、当前行为、最小复现场景、截图/日志、影响范围。  
**PR 描述建议包含**：变更类型（feat/fix/docs/refactor/perf/chore）、详细说明、测试方式、相关 Issue 号、破坏性变更声明（如有）。

## 分支策略

- `main`：稳定分支
- `feature/<topic>`：功能开发
- `fix/<bug>`：缺陷修复
- 大型重构：`refactor/<scope>`

## 新增配置项流程

1. 在 `graphrag/config/settings.py` 添加字段与默认值（必要时分类注释）。
2. 更新 `.env` 示例（添加注释，说明用途与默认）。
3. 若属于可选特性：在 README(en/zh) Feature Switch 表 & Features 文档“可选特性”同步。
4. 若需出现在 `/diagnostics`：更新对应 diagnostics 组装逻辑。
5. 编写最小使用说明或调参说明。
6. 若会影响排序或权重：更新调参表格。

## 代码风格与质量

- Python：保持与现有文件一致的命名与导入顺序
- 避免过度抽象；优先可读性 > 过早通用化
- 减少嵌套：必要时提取小函数（尤其 LLM 解析 / 图构建阶段）
- 所有 catch 均应记录上下文（至少 debug 级别），避免静默吞错

## 测试与诊断脚本

| 类型 | 命令 | 说明 |
|------|------|------|
| 环境变量同步 | `python scripts/check_env_sync.py` | 列出缺失 / 额外 / 默认覆盖 |
| 依赖导入 | `python scripts/check_requirements_imports.py` | 检查 requirements 中依赖可导入性 |
| 权重/索引 dump | `python -m graphrag.diagnostics.dump` | 输出权重、embedding 维度、索引健康 |
| 排序预览 | `curl -X POST /ranking/preview` | 查看 composite score 组成 |

## 常见贡献类型建议

| 类型 | 说明 | 建议切入点 |
|------|------|------------|
| 文档优化 | 修正文档、补充示例、双语同步 | README / Features / CONTRIBUTING |
| 新检索信号 | 新的得分或扩展策略 | `retriever/` 模块扩展并加入 diagnostics |
| 真实 Rerank | Cross-Encoder 模型接入 | 新建 `rerank.py` 内实际模型调用 |
| 图算法增强 | PageRank / 中心性统计 | `utils/graph_rank.py` 扩展缓存策略 |
| 噪声策略 | 更智能实体过滤 | `ingest` 中实体后处理阶段插入 |
| 监控扩展 | 指标上报/Prometheus | 新增 middleware 或独立 metrics 模块 |

## 发布与版本

- 采用轻量语义化（可在日后引入 CHANGELOG）：`feat:` / `fix:` / `docs:` 前缀有助于整理版本摘要
- 首次外部发布后建议引入 `CHANGELOG.md`

---

## Contributing Guide (English)

## Quick Navigation

- [Code of Conduct](#code-of-conduct)
- [Dev Environment](#dev-environment)
- [Pre-commit Checklist](#pre-commit-checklist)
- [Issue / PR Guidelines](#issue--pr-guidelines)
- [Branching](#branching)
- [Config Addition Workflow](#config-addition-workflow)
- [Code Style](#code-style)
- [Diagnostics & Scripts](#diagnostics--scripts)
- [Contribution Ideas](#contribution-ideas)
- [Release & Versioning](#release--versioning)

## Code of Conduct

Be respectful, concise, actionable. No harassment, discrimination, illegal content.

## Dev Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn graphrag.service.server:app --reload --port 8000
```

Optional sanity checks:

```bash
python scripts/check_env_sync.py
python scripts/check_requirements_imports.py
python -m graphrag.diagnostics.dump
```

## Pre-commit Checklist

- [ ] New / changed env vars reflected in: `settings.py` + `.env` + both READMEs + feature docs
- [ ] Added minimal test or reproducible script for critical logic
- [ ] Updated `/diagnostics` if new runtime flags / counters
- [ ] Bilingual docs synced (if user-visible)
- [ ] Removed stray debug prints
- [ ] Passed lint / import verification scripts

## Issue / PR Guidelines

Provide: context, expected vs actual, reproduction steps, logs, scope impact.  
PR description: change type (feat/fix/docs/refactor/perf/chore), details, test method, linked issues, breaking changes.

## Branching

- `main` (stable)
- `feature/<topic>`
- `fix/<bug>`
- `refactor/<scope>`

## Config Addition Workflow

1. Add field in `graphrag/config/settings.py`.
2. Mirror in `.env` example with comment.
3. Document in both READMEs + feature guides (and Feature Switch tables if toggleable).
4. Expose in `/diagnostics` if runtime-inspectable.
5. Provide tuning / usage guidance.
6. Update tuning / weights tables if ranking affected.

## Code Style

- Follow existing naming/import patterns
- Small focused functions > deep nesting
- Log context on exceptions; avoid silent except
- Prefer clarity; postpone abstraction until repeated thrice

## Diagnostics & Scripts

| Type | Command | Purpose |
|------|---------|---------|
| Env sync | `python scripts/check_env_sync.py` | Compare `.env` vs settings definitions |
| Imports | `python scripts/check_requirements_imports.py` | Validate importability |
| Dump | `python -m graphrag.diagnostics.dump` | Show weights & embedding/index dims |
| Ranking preview | `curl -X POST /ranking/preview` | Inspect composite score breakdown |

## Contribution Ideas

| Area | Description | Entry Point |
|------|-------------|------------|
| Docs | Improve bilingual docs & examples | README / Features |
| Sparse/Hybrid | New retrieval signals or weighting schemas | `retriever/` |
| True Rerank | Cross-Encoder integration | `retriever/rerank.py` |
| Graph Algorithms | PageRank / centrality cache | `utils/graph_rank.py` |
| Noise Filtering | Advanced entity heuristics | ingest entity post-process |
| Observability | Prometheus / metrics export | new middleware |

## Release & Versioning

Lightweight semantic commit prefixes help future CHANGELOG generation. Introduce `CHANGELOG.md` after first external release.

---

> Keep this file updated when adding features / flags / diagnostics fields.

---

### 版权 / Copyright

版权所有 © 2025 EvolutionAI Studio（All Rights Holder: Johnny Zhang）  
社区共同参与改进，提交历史可见贡献来源。再分发请保留 `LICENSE` 与 `NOTICE`。  
Copyright © 2025 EvolutionAI Studio (All Rights Holder: Johnny Zhang).  
Retain `LICENSE` and `NOTICE` in any redistribution.
