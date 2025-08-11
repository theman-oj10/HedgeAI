# 🤖 Hedge AI

**A Personal AI Hedge Fund for Retail Investors**

We built an intelligent investment platform powered by AI agents modeled after legendary investors like Warren Buffett, Cathie Wood, and Michael Burry. This system democratizes institutional-grade investment analysis for retail investors through AI-driven debates and automated portfolio management.

## 🎯 Overview

Our AI hedge fund creates a virtual boardroom where legendary investor personalities debate investment decisions in real-time. Each agent brings their unique investment philosophy and analytical approach, while a Portfolio Manager agent orchestrates the discussions and synthesizes insights into actionable trading decisions.

### 🧠 AI Investment Agents

**Legendary Investor Personalities:**
- **Warren Buffett Agent** - The Oracle of Omaha: Value investing focused on wonderful companies at fair prices
- **Cathie Wood Agent** - Innovation Catalyst: Growth investing in disruptive technologies and exponential trends  
- **Michael Burry Agent** - The Contrarian: Deep value hunting with contrarian market insights

**Specialized Analysis Agents:**
- **Valuation Agent** - Intrinsic value calculations and DCF modeling
- **Sentiment Agent** - Market sentiment and news analysis
- **Fundamentals Agent** - Financial statement and ratio analysis
- **Technical Agent** - Chart patterns and technical indicators
- **Risk Manager** - Portfolio risk assessment and position sizing

### 🔧 Professional-Grade Tools

Each agent is equipped with institutional-quality analysis tools:

- **📊 Fundamental Analysis** - Company performance evaluation, financial health assessment, and valuation modeling
- **📰 Market News & Sentiment** - Real-time news analysis, social media sentiment, and market mood indicators  
- **🌍 Macroeconomic Indicators** - Economic trends, interest rates, inflation data, and sector rotation analysis
- **📈 Technical Analysis** - Chart patterns, momentum indicators, and trend analysis
- **⚖️ Risk Management** - Position sizing, portfolio diversification, and risk-adjusted returns

## 🏛️ The AI Board of Directors

At the heart of our system is the **Portfolio Manager Agent** - acting as the moderator and decision-maker in every investment discussion. This agent:

- **Orchestrates Debates** - Facilitates structured discussions between investor agents
- **Synthesizes Insights** - Combines different investment perspectives into coherent strategies
- **Makes Final Decisions** - Weighs all arguments and executes trades based on consensus
- **Manages Risk** - Ensures portfolio balance and adherence to risk parameters

### 📅 Quarterly Strategic Reviews

Every quarter, the AI Board convenes for comprehensive portfolio reviews:
- **Performance Evaluation** - Analyzing returns, risk metrics, and benchmark comparisons
- **Market Reassessment** - Updating market outlook and identifying new opportunities
- **Portfolio Rebalancing** - Optimizing asset allocation and position sizing
- **Strategy Refinement** - Adapting investment approach based on market conditions

## 🚀 Technology Stack

**Development Tools:**
- **Windsurf** - AI-powered development environment
- **Claude Code** - Advanced code generation and debugging

**AI Infrastructure:**
- **OpenAI APIs** - GPT-4 for sophisticated reasoning and analysis
- **Cerebras APIs** - High-performance inference for scalable processing
- **Custom LLM Pipeline** - Optimized for financial analysis and decision-making

**Backtesting Framework:**
- **Custom Simulation Engine** - Historical performance testing from 2022-2025
- **Risk-Adjusted Metrics** - Sharpe ratio, maximum drawdown, and volatility analysis
- **Benchmark Comparison** - Performance vs SPY and other market indices

## 📈 Performance Results

Our AI hedge fund has demonstrated strong performance in backtesting:
- **Outperformed SPY** - Beat the S&P 500 benchmark over the 2022-2025 period
- **Risk Management** - Maintained disciplined position sizing and diversification
- **Market Adaptability** - Successfully navigated different market conditions and volatility regimes

<img width="1042" alt="AI Hedge Fund Dashboard" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application (NEW!)](#️-web-application)
- [Contributing](#contributing)
- [Feature Requests](#feature-requests)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.



### 2. Set Up Your API Keys

Create a `.env` file for your API keys:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-api-key

# For running LLMs hosted by groq (deepseek, llama3, etc.)
GROQ_API_KEY=your-groq-api-key

# For getting financial data to power the hedge fund
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**Important**: You must set at least one LLM API key (`OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work. 

**Financial Data**: Data for AAPL, GOOGL, MSFT, NVDA, and TSLA is free and does not require an API key. For any other ticker, you will need to set the `FINANCIAL_DATASETS_API_KEY` in the .env file.

## How to Run

### ⌨️ Command Line Interface

For users who prefer working with command line tools, you can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

Choose one of the following installation methods:

```

#### Running the AI Hedge Fund (with Poetry)
```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

#### Running the AI Hedge Fund (with Docker)
```bash
# Navigate to the docker directory first
cd docker

# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA main
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
# With Poetry:
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --ollama main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --ollama main
```

You can also specify a `--show-reasoning` flag to print the reasoning of each agent to the console.

```bash
# With Poetry:
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --show-reasoning

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --show-reasoning main

# On Windows:
run.bat --ticker AAPL, MSFT,NVDA --show-reasoning main
```

You can optionally specify the start and end dates to make decisions for a specific time period.

```bash
# With Poetry:
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 

# With Docker (from docker/ directory):
# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01 main
```

#### Running the Backtester (with Poetry)
```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

#### Running the Backtester (with Docker)
```bash
# Navigate to the docker directory first
cd docker

# On Linux/Mac:
./run.sh --ticker AAPL,MSFT,NVDA backtest

# On Windows:
run.bat --ticker AAPL,MSFT,NVDA backtest
```

### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. **This is recommended for most users, especially those who prefer visual interfaces over command line tools.**

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />

#### For Mac/Linux:
```bash
cd app && ./run.sh
```

If you get a "permission denied" error, run this first:
```bash
cd app && chmod +x run.sh && ./run.sh
```

#### For Windows:
```bash
# Go to /app directory
cd app

# Run the app
\.run.bat
```

**That's it!** These scripts will:
1. Check for required dependencies (Node.js, Python, Poetry)
2. Install all dependencies automatically  
3. Start both frontend and backend services
4. **Automatically open your web browser** to the application


#### Detailed Setup Instructions

For detailed setup instructions, troubleshooting, and advanced configuration options, see:
- [Full-Stack App Documentation](./app/README.md)
- [Frontend Documentation](./app/frontend/README.md)  
- [Backend Documentation](./app/backend/README.md)


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
