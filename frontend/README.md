# GoodKnight Chess Engine Frontend

A React-based web interface for playing against the GoodKnight chess engine powered by AWS Lambda.

## Features

- **Interactive Chess Board**: Drag-and-drop interface for making moves
- **Play vs Engine**: Play as White or Black against your custom chess engine
- **Move History**: Track all moves played during the game
- **Engine Info**: Display engine thinking time and status
- **Game Controls**: Start new games and switch colors
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Node.js 18+ and npm
- AWS Lambda function deployed (see parent directory for backend setup)

## Setup

1. Install dependencies:
```bash
npm install
```

2. Configure the API endpoint:
```bash
cp .env.example .env
```

3. Edit `.env` and set your Lambda API endpoint:
```
VITE_CHESS_API_ENDPOINT=https://your-lambda-url.execute-api.region.amazonaws.com/prod/move
```

## Development

Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

## Building for Production

Build the app:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── ChessGame.jsx       # Main game component
│   ├── ChessGame.css       # Game styling
│   ├── MoveHistory.jsx     # Move history display
│   ├── MoveHistory.css     # Move history styling
│   ├── EngineInfo.jsx      # Engine status display
│   └── EngineInfo.css      # Engine info styling
├── services/
│   └── chessApi.js         # Lambda API integration
├── App.jsx                 # Root component
├── App.css                 # App styling
├── index.css               # Global styles
└── main.jsx                # Entry point
```

## Technologies Used

- **React** - UI framework
- **Vite** - Build tool and dev server
- **chess.js** - Chess game logic and validation
- **react-chessboard** - Interactive chess board component

## Lambda API Integration

The frontend expects your Lambda function to accept POST requests with this format:

**Request:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response:**
```json
{
  "move": "e2e4"
}
```

The move should be in UCI format (e.g., "e2e4", "e7e5", "e1g1" for castling, "e7e8q" for promotion).

## Deployment

### Deploy to AWS S3 + CloudFront

1. Build the app:
```bash
npm run build
```

2. Upload the `dist/` folder to an S3 bucket configured for static website hosting

3. (Optional) Set up CloudFront for CDN distribution

### Deploy to Vercel/Netlify

These platforms can auto-detect Vite projects. Just connect your repository and they'll handle the build.

## Troubleshooting

**"API endpoint not configured" warning**
- Make sure you've created a `.env` file with `VITE_CHESS_API_ENDPOINT` set to your Lambda URL
- Environment variables must start with `VITE_` to be exposed to the frontend
- Restart the dev server after changing environment variables

**CORS errors**
- Ensure your Lambda function returns proper CORS headers:
  ```python
  'headers': {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',  # Or your specific domain
      'Access-Control-Allow-Headers': 'Content-Type',
      'Access-Control-Allow-Methods': 'POST, OPTIONS'
  }
  ```

**Engine not responding**
- Check the browser console for error messages
- Verify your Lambda function is deployed and the URL is correct
- Test the Lambda endpoint directly using curl or Postman

## License

MIT
