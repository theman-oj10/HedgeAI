import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import fs from 'fs'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api/backtest-data': {
        bypass: (req, res) => {
          const dataPath = '/Users/mingjun/HedgeAI/backtest_results/backtest_data.json';
          try {
            const data = fs.readFileSync(dataPath, 'utf-8');
            res.setHeader('Content-Type', 'application/json');
            res.end(data);
          } catch (error) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: 'Failed to read data file' }));
          }
          return null;
        }
      }
    }
  }
})
