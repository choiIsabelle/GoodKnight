import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  define: {
    'process.env.VITE_CHESS_API_ENDPOINT': JSON.stringify(process.env.VITE_CHESS_API_ENDPOINT)
  },
  server: {
    proxy: {
      '/lambda': {
        target: 'http://localhost:9000',
        changeOrigin: true,
        rewrite: (path) =>
          path.replace(/^\/lambda/, '/2015-03-31/functions/function/invocations'),
      },
    },
  },
})
