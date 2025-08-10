# AI Investment Agents - Setup Guide

## System Requirements

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd ai-hedge-fund/hedge_ai
```

### 2. Install Dependencies

#### Option A: Using pip (recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Using Poetry (if you prefer)
```bash
poetry install
```

### 3. Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
# Copy the example environment file
cp .env.example .env
```

Edit `.env` and add your API keys:
```
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LLM Provider (openai or anthropic)
LLM_PROVIDER=openai
```

### 4. Verify Installation

Test the installation with a simple portfolio analysis:

```bash
# From the hedge_ai directory
cd src
python -m portfolio_debate custom AAPL:1.0
```

## Usage Examples

### Single Stock Analysis
```bash
cd src
python -m consensus_debate AAPL
```

### Portfolio Analysis
```bash
cd src
# Predefined portfolio
python -m portfolio_debate tech_growth

# Custom portfolio
python -m portfolio_debate custom AAPL:0.4 MSFT:0.3 GOOGL:0.3
```

### JSON Output
The system automatically generates JSON output with trading actions and saves it to a file.

## Project Structure

```
hedge_ai/
├── src/
│   ├── agents/
│   │   ├── warren_buffett_agent.py
│   │   ├── cathie_wood_agent.py
│   │   └── moderator_agent.py
│   ├── portfolio_config.py
│   ├── portfolio_debate.py
│   ├── consensus_debate.py
│   ├── utils.py
│   └── main.py
├── portfolios/
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### Import Errors
If you encounter import errors, make sure you're running commands from the correct directory:
- For portfolio analysis: Run from `hedge_ai/src/`
- Use `python -m module_name` format for better module resolution

### API Key Issues
- Ensure your `.env` file is in the project root
- Verify API keys are valid and have sufficient credits
- Check that `LLM_PROVIDER` matches your available API key

### Path Issues
The system uses relative imports and should work on any platform. If you encounter path issues:
1. Ensure you're running from the `src/` directory
2. Use the `-m` flag when running Python modules
3. Check that all files are in the correct directory structure

## Platform Compatibility

This system is designed to work on:
- ✅ Windows
- ✅ macOS  
- ✅ Linux

All path handling uses `os.path` for cross-platform compatibility.
