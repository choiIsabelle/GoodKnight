import { useState, useCallback, useEffect } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import { getBestMove, isApiConfigured } from '../services/chessApi';
import MoveHistory from './MoveHistory';
import EngineInfo from './EngineInfo';
import About from './About';
import './ChessGame.css';

export default function ChessGame() {
  const [game, setGame] = useState(new Chess());
  const [position, setPosition] = useState(game.fen());
  const [moveHistory, setMoveHistory] = useState([]);
  const [engineThinking, setEngineThinking] = useState(false);
  const [engineInfo, setEngineInfo] = useState(null);
  const [playerColor, setPlayerColor] = useState('white');
  const [gameOver, setGameOver] = useState(false);
  const [gameResult, setGameResult] = useState('');
  const [apiConfigured] = useState(isApiConfigured());

  // Check for game over conditions
  const checkGameOver = useCallback((chess) => {
    if (chess.isGameOver()) {
      let result = '';
      if (chess.isCheckmate()) {
        result = `Checkmate! ${chess.turn() === 'w' ? 'Black' : 'White'} wins!`;
      } else if (chess.isDraw()) {
        if (chess.isStalemate()) {
          result = 'Draw by stalemate';
        } else if (chess.isThreefoldRepetition()) {
          result = 'Draw by threefold repetition';
        } else if (chess.isInsufficientMaterial()) {
          result = 'Draw by insufficient material';
        } else {
          result = 'Draw';
        }
      }
      setGameOver(true);
      setGameResult(result);
      return true;
    }
    return false;
  }, []);

  // Make engine move
  const makeEngineMove = useCallback(async () => {
    console.log('makeEngineMove called', { gameOver, turn: game.turn(), playerColor: playerColor[0] });

    if (gameOver) return;

    const currentGame = new Chess(game.fen());

    if (currentGame.turn() === playerColor[0]) {
      console.log('Skipping engine move - player turn');
      return; // It's the player's turn
    }

    if (!apiConfigured) {
      console.warn('API endpoint not configured');
      return;
    }

    console.log('Engine making move...');
    setEngineThinking(true);
    setEngineInfo({ status: 'thinking...', thinkingTime: null });

    try {
      const { move: uciMove, thinkingTime } = await getBestMove(currentGame.fen());
      console.log('Engine move received:', uciMove);

      // Convert UCI format to chess.js move
      const move = currentGame.move({
        from: uciMove.substring(0, 2),
        to: uciMove.substring(2, 4),
        promotion: uciMove.length > 4 ? uciMove[4] : undefined,
      });

      if (move) {
        setGame(currentGame);
        setPosition(currentGame.fen());
        setMoveHistory(prev => [...prev, {
          moveNumber: Math.floor(currentGame.moveNumber()),
          white: currentGame.turn() === 'b' ? move.san : prev[prev.length - 1]?.white,
          black: currentGame.turn() === 'w' ? move.san : null,
        }]);
        setEngineInfo({
          status: 'ready',
          thinkingTime: thinkingTime.toFixed(0),
          lastMove: move.san,
        });

        checkGameOver(currentGame);
      }
    } catch (error) {
      console.error('Engine error:', error);
      setEngineInfo({
        status: 'error',
        error: error.message,
      });
    } finally {
      setEngineThinking(false);
    }
  }, [game, playerColor, gameOver, apiConfigured, checkGameOver]);

  // Trigger engine move when it's engine's turn
  useEffect(() => {
    if (!gameOver && game.turn() !== playerColor[0] && !engineThinking) {
      // Small delay to make it feel more natural
      const timer = setTimeout(() => {
        makeEngineMove();
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [game, playerColor, engineThinking, gameOver, makeEngineMove]);

  // Handle player move
  const onDrop = useCallback(({ sourceSquare, targetSquare, piece }) => {
    if (engineThinking || gameOver) {
      return false;
    }

    if (game.turn() !== playerColor[0]) {
      return false;
    }

    const currentGame = new Chess(game.fen());

    try {
      const move = currentGame.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q', // Always promote to queen for simplicity
      });

      if (move) {
        setGame(currentGame);
        setPosition(currentGame.fen());

        // Update move history
        if (playerColor === 'white') {
          setMoveHistory(prev => [...prev, {
            moveNumber: currentGame.moveNumber(),
            white: move.san,
            black: null,
          }]);
        } else {
          setMoveHistory(prev => {
            const newHistory = [...prev];
            if (newHistory.length === 0 || newHistory[newHistory.length - 1].black !== null) {
              newHistory.push({
                moveNumber: currentGame.moveNumber(),
                white: null,
                black: move.san,
              });
            } else {
              newHistory[newHistory.length - 1].black = move.san;
            }
            return newHistory;
          });
        }

        if (!checkGameOver(currentGame)) {
          setEngineInfo({ status: 'preparing...', thinkingTime: null });
        }

        return true;
      }
    } catch (error) {
      console.error('Move error:', error);
      return false;
    }

    return false;
  }, [game, playerColor, engineThinking, gameOver, checkGameOver]);

  // Control which pieces are draggable
  const isDraggablePiece = useCallback(({ piece, square }) => {
    if (engineThinking || gameOver) {
      return false;
    }

    // Get the actual piece from the current position
    const pieceOnSquare = game.get(square);
    if (!pieceOnSquare) {
      return false;
    }

    // Only allow dragging pieces of the player's color
    const pieceColor = pieceOnSquare.color; // 'w' or 'b'
    const canDrag = pieceColor === playerColor[0] && game.turn() === playerColor[0];

    return canDrag;
  }, [playerColor, game, engineThinking, gameOver]);

  // Reset game
  const resetGame = () => {
    const newGame = new Chess();
    setGame(newGame);
    setPosition(newGame.fen());
    setMoveHistory([]);
    setEngineInfo(null);
    setGameOver(false);
    setGameResult('');
  };

  // Switch player color
  const switchColor = () => {
    setPlayerColor(prev => prev === 'white' ? 'black' : 'white');
    resetGame();
  };

  return (
    <div className="chess-game">
      <div className="game-header">
        <h1>GoodKnight Chess Engine</h1>
        {!apiConfigured && (
          <div className="api-warning">
            API endpoint not configured. Please set VITE_CHESS_API_ENDPOINT in .env file.
          </div>
        )}
      </div>

      <div className="game-container">
        <div className="about-section">
          <About />
        </div>

        <div className="board-section">
          <div className="board-controls">
            <button onClick={resetGame} disabled={engineThinking}>
              New Game
            </button>
            <button onClick={switchColor} disabled={engineThinking}>
              Play as {playerColor === 'white' ? 'Black' : 'White'}
            </button>
            <span className="turn-indicator">
              {gameOver
                ? 'Game Over'
                : `${game.turn() === 'w' ? 'White' : 'Black'} to move`}
            </span>
          </div>

          <div className="chessboard-wrapper">
            <Chessboard
              options={{
                position: position,
                onPieceDrop: onDrop,
                canDragPiece: isDraggablePiece,
                boardOrientation: playerColor,
                boardStyle: {
                  borderRadius: '4px',
                  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
                },
                showNotation: true,
                darkSquareStyle: { backgroundColor: '#b58863' },
                lightSquareStyle: { backgroundColor: '#f0d9b5' },
              }}
            />
          </div>

          {gameOver && (
            <div className="game-result">
              {gameResult}
            </div>
          )}
        </div>

        <div className="info-section">
          <EngineInfo
            engineInfo={engineInfo}
            engineThinking={engineThinking}
            apiConfigured={apiConfigured}
          />
          <MoveHistory moves={moveHistory} />
        </div>
      </div>
    </div>
  );
}
