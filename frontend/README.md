# Evolution GraphRAG Frontend (React + Vite)

开发说明：

- 默认后端地址：`http://localhost:8010`（可通过 `.env.local` 覆盖）
- 运行：

```bash
npm install
npm run dev
```

- 可选：创建 `.env.local` 指定后端地址

```bash
VITE_API_BASE=http://localhost:8010
```

- 生产构建：

```bash
npm run build
npm run preview
```

- 开发代理：`vite.config.ts` 已将 `/query`、`/ingest`、`/docs`、`/config` 等路径代理到 `VITE_API_BASE`。
