# GoodKnight ♟️

A hybrid chess engine built at [Chess Hacks](https://chesshacks.dev/) at the University of Waterloo, combining classical chess engine techniques with modern AI evaluation.

**[🎮 Try it live!](https://choiIsabelle.github.io/GoodKnight)**

## Overview

GoodKnight merges traditional chess engine approaches with machine learning to create a powerful chess AI:

- **Classical Engine**: Implements alpha-beta pruning for efficient move tree search
- **AI Evaluation**: Uses a PyTorch-based neural network to evaluate board positions
- **Serverless Architecture**: Runs on AWS Lambda for scalable, cost-effective deployment
- **Modern Frontend**: React-based web interface with real-time game analysis

## Architecture

```
┌─────────────────┐
│  React Frontend │  ← https://choiIsabelle.github.io/GoodKnight
└────────┬────────┘
         │ POST /move
         ▼
┌─────────────────┐
│  AWS Lambda     │  ← Chess engine backend
│  (Docker)       │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────┐
│ Alpha-  │ │ PyTorch  │
│ Beta    │ │ Model    │
│ Pruning │ │ Eval     │
└─────────┘ └──────────┘
```

## How It Works

1. **Move Generation**: Classical chess engine generates legal moves
2. **Tree Search**: Alpha-beta pruning explores the game tree efficiently
3. **Position Evaluation**: PyTorch model evaluates leaf nodes
4. **Best Move Selection**: Engine returns the highest-scoring move

The hybrid approach combines the speed of classical algorithms with the positional understanding of neural networks.

## Submodules

This project uses the following submodules:

- **[GoodKnightCommon](https://github.com/EricHayter/GoodKnightCommon)** - Shared utilities and board representation
- **[GoodKnightModel](https://github.com/mchoi-cs/GoodKnightModel)** - PyTorch neural network for position evaluation

## Getting Started

### Prerequisites

- Docker
- Node.js 18+ (for frontend development)
- Python 3.12 (for local development without Docker)

### Clone with Submodules

```bash
git clone --recursive https://github.com/choiIsabelle/GoodKnight.git
cd GoodKnight
```

If you already cloned without `--recursive`:
```bash
git submodule update --init --recursive
```

## Running Locally

### Option 1: Docker (Recommended)

Build and run the Lambda function container:

```bash
# Build the Docker image
docker build -t goodknight-lambda .

# Run the container locally (Lambda Runtime Interface Emulator)
docker run -p 9000:8080 goodknight-lambda
```

Test the engine:
```bash
curl -X POST "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

### Option 2: Local Python Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r src/GoodKnightCommon/requirements.txt
pip install -r src/GoodKnightModel/requirements-cpu.txt

# Run the Lambda handler locally
python lambda_handler.py
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server (proxies /lambda to localhost:9000)
npm run dev

# Build for production
npm run build

# Deploy to GitHub Pages
npm run deploy
```

**Environment Variables:**

The frontend uses `VITE_CHESS_API_ENDPOINT` to configure the API endpoint:

- **Local development**: Uses `/lambda` (proxied to `http://localhost:9000` via Vite)
- **Production**: Set via environment variable during build:
  ```bash
  VITE_CHESS_API_ENDPOINT=https://your-lambda-url.on.aws/ npm run build
  ```

## Project Structure

```
GoodKnight/
├── frontend/              # React web interface
│   ├── src/
│   │   ├── components/    # React components
│   │   └── services/      # API client
│   └── vite.config.js     # Vite configuration
├── src/                   # Chess engine source
│   ├── GoodKnightCommon/  # Submodule: shared utilities
│   ├── GoodKnightModel/   # Submodule: PyTorch model
│   └── main.py           # Engine entry point
├── lambda_handler.py     # AWS Lambda handler
├── Dockerfile            # Lambda container image
└── requirements.txt      # Python dependencies
```

## Deployment

### AWS Lambda

1. Build the Docker image:
   ```bash
   docker build -t goodknight-lambda .
   ```

2. Push to Amazon ECR and deploy to Lambda (see [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html))

3. Configure Lambda:
   - Memory: 1024 MB+ (higher = faster execution)
   - Timeout: 3 minutes
   - Function URL: Enable for public access

### GitHub Pages

The frontend automatically deploys to GitHub Pages via the `gh-pages` branch.

Set up GitHub Actions with your Lambda URL:
```yaml
- name: Build and Deploy
  env:
    VITE_CHESS_API_ENDPOINT: https://your-lambda-url.lambda-url.us-east-1.on.aws/
  run: |
    cd frontend
    npm install
    npm run deploy
```

## Technologies

- **Backend**: Python, PyTorch, AWS Lambda
- **Frontend**: React, Vite, react-chessboard
- **Chess Logic**: chess.js, python-chess
- **Infrastructure**: Docker, GitHub Pages

## Contributors

Built during Chess Hacks at the University of Waterloo.

- [@choiIsabelle](https://github.com/choiIsabelle)
- [@EricHayter](https://github.com/EricHayter)
- [@mchoi-cs](https://github.com/mchoi-cs)

## License

MIT
