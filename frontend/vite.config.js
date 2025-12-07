import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env file based on `mode` in the current working directory.
  // Set the third parameter to '' to load all env regardless of the `VITE_` prefix.
  const env = loadEnv(mode, process.cwd(), '')

  // Environment variables from the actual environment (e.g., GitHub workflows)
  // take precedence over .env files
  const VITE_CHESS_API_ENDPOINT = process.env.VITE_CHESS_API_ENDPOINT || env.VITE_CHESS_API_ENDPOINT

  return {
    base: "/GoodKnight",
    plugins: [react()],
    define: {
      // Explicitly define env variables, prioritizing process.env over .env files
      'import.meta.env.VITE_CHESS_API_ENDPOINT': JSON.stringify(VITE_CHESS_API_ENDPOINT),
    },
    server: {
      proxy: {
        '/lambda': {
          target: 'http://localhost:9000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/lambda/, '/2015-03-31/functions/function/invocations'),
        }
      }
    }
  }
})
