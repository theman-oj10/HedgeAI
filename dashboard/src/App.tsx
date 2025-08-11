import { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './App.css';

interface Position {
  shares: number;
  cost_basis: number;
  current_value: number;
}

interface Trade {
  action: string;
  quantity: number;
  price: number;
  cost?: number;
  proceeds?: number;
  realized_gain?: number;
  reason: string;
}

interface Iteration {
  session_id: string;
  portfolio_name: string;
  timestamp: string;
  iteration_date: string;
  portfolio_value: number;
  cash: number;
  positions: Record<string, Position>;
  executed_trades: Record<string, Trade>;
  current_prices: Record<string, number>;
}

interface BacktestData {
  iterations: Iteration[];
  metadata: {
    last_updated: string;
    portfolio_name: string;
    start_date: string;
    end_date: string;
    initial_capital: number;
    trading_frequency: string;
    current_session_id: string;
  };
}

function App() {
  const [data, setData] = useState<BacktestData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  const fetchData = async () => {
    try {
      const response = await fetch('/api/backtest-data');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const jsonData = await response.json();
      setData(jsonData);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  if (error) {
    return (
      <div className="dashboard">
        <h1>Backtest Dashboard</h1>
        <div className="error">Error: {error}</div>
        <p>Retrying in 5 seconds...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="dashboard">
        <h1>Backtest Dashboard</h1>
        <div className="loading">Loading...</div>
      </div>
    );
  }

  const portfolioValueData = data.iterations.map(iter => ({
    date: new Date(iter.iteration_date).toLocaleDateString('en-US', { month: 'short', year: 'numeric' }),
    value: Math.round(iter.portfolio_value),
    cash: Math.round(iter.cash)
  }));

  const currentIteration = data.iterations[data.iterations.length - 1];
  const positionsData = currentIteration ? Object.entries(currentIteration.positions).map(([ticker, position]) => ({
    ticker,
    shares: position.shares,
    value: Math.round(position.current_value),
    costBasis: Math.round(position.cost_basis),
    gain: Math.round(position.current_value - position.cost_basis)
  })) : [];

  const tradesData = currentIteration ? Object.entries(currentIteration.executed_trades).map(([ticker, trade]) => ({
    ticker,
    action: trade.action,
    quantity: trade.quantity,
    price: trade.price.toFixed(2),
    amount: trade.cost ? Math.round(trade.cost) : Math.round(trade.proceeds || 0),
    gain: trade.realized_gain ? Math.round(trade.realized_gain) : null
  })) : [];

  const chartColors = {
    line1: darkMode ? '#a5a1ff' : '#8884d8',
    line2: darkMode ? '#82ca9d' : '#82ca9d',
    grid: darkMode ? '#404040' : '#e0e0e0',
    text: darkMode ? '#b0b0b0' : '#666666'
  };

  return (
    <div className="dashboard">
      <header>
        <h1>Backtest Dashboard - {data.metadata.portfolio_name}</h1>
        <div className="metadata">
          <span>Session: {data.metadata.current_session_id}</span>
          <span>Last Update: {lastUpdate.toLocaleTimeString()}</span>
          <span className="update-indicator">🔄 Auto-updating every 5s</span>
          <button className="dark-mode-toggle" onClick={toggleDarkMode}>
            {darkMode ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <h3>Initial Capital</h3>
          <p className="stat-value">${data.metadata.initial_capital.toLocaleString()}</p>
        </div>
        <div className="stat-card">
          <h3>Current Value</h3>
          <p className="stat-value">${currentIteration ? Math.round(currentIteration.portfolio_value).toLocaleString() : '-'}</p>
        </div>
        <div className="stat-card">
          <h3>Total Return</h3>
          <p className="stat-value">
            {currentIteration ? 
              ((currentIteration.portfolio_value / data.metadata.initial_capital - 1) * 100).toFixed(2) + '%' 
              : '-'}
          </p>
        </div>
        <div className="stat-card">
          <h3>Cash Available</h3>
          <p className="stat-value">${currentIteration ? Math.round(currentIteration.cash).toLocaleString() : '-'}</p>
        </div>
      </div>

      <div className="chart-section">
        <h2>Portfolio Value Over Time</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={portfolioValueData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis dataKey="date" stroke={chartColors.text} />
            <YAxis stroke={chartColors.text} />
            <Tooltip formatter={(value) => `$${value.toLocaleString()}`} contentStyle={{ backgroundColor: darkMode ? '#2a2a2a' : '#fff', border: `1px solid ${chartColors.grid}` }} />
            <Legend wrapperStyle={{ color: chartColors.text }} />
            <Line type="monotone" dataKey="value" stroke={chartColors.line1} name="Portfolio Value" />
            <Line type="monotone" dataKey="cash" stroke={chartColors.line2} name="Cash" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="positions-section">
        <h2>Current Positions</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={positionsData}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis dataKey="ticker" stroke={chartColors.text} />
            <YAxis stroke={chartColors.text} />
            <Tooltip formatter={(value) => `$${value.toLocaleString()}`} contentStyle={{ backgroundColor: darkMode ? '#2a2a2a' : '#fff', border: `1px solid ${chartColors.grid}` }} />
            <Legend wrapperStyle={{ color: chartColors.text }} />
            <Bar dataKey="value" fill={chartColors.line1} name="Current Value" />
            <Bar dataKey="costBasis" fill={chartColors.line2} name="Cost Basis" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="tables-section">
        <div className="table-container">
          <h3>Positions Detail</h3>
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Shares</th>
                <th>Current Value</th>
                <th>Cost Basis</th>
                <th>Gain/Loss</th>
              </tr>
            </thead>
            <tbody>
              {positionsData.map(pos => (
                <tr key={pos.ticker}>
                  <td>{pos.ticker}</td>
                  <td>{pos.shares}</td>
                  <td>${pos.value.toLocaleString()}</td>
                  <td>${pos.costBasis.toLocaleString()}</td>
                  <td className={pos.gain >= 0 ? 'positive' : 'negative'}>
                    ${pos.gain.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-container">
          <h3>Recent Trades</h3>
          <table>
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Action</th>
                <th>Quantity</th>
                <th>Price</th>
                <th>Amount</th>
                <th>Realized Gain</th>
              </tr>
            </thead>
            <tbody>
              {tradesData.map((trade, idx) => (
                <tr key={`${trade.ticker}-${idx}`}>
                  <td>{trade.ticker}</td>
                  <td className={trade.action === 'buy' ? 'buy' : 'sell'}>{trade.action.toUpperCase()}</td>
                  <td>{trade.quantity}</td>
                  <td>${trade.price}</td>
                  <td>${trade.amount.toLocaleString()}</td>
                  <td className={trade.gain !== null ? (trade.gain >= 0 ? 'positive' : 'negative') : ''}>
                    {trade.gain !== null ? `$${trade.gain.toLocaleString()}` : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default App
