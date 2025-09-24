import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const api = env.VITE_API_BASE || 'http://localhost:8010'
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/health': api,
        '/ingest': api,
        '/query': api,
        '/diagnostics': api,
        '/cache': api,
        '/ranking': api,
        '/docs': api,
        '/config': api,
        '/qa': api,
        '/prompts': api
      }
    }
  }
})
