from chess import Move, Board
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'chess-hacks-training-main'))
sys.path.insert(0, str(Path(__file__).parent / 'GoodKnightCommon'))

from chess_cnn import create_model
from fen_to_tensor import get_tensor_bytes_from_fen

weights_path = Path(__file__).parent / 'weights' / 'weights.pth'
print(f"Loading model weights from {weights_path}...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model and load weights
model = create_model(num_filters=32, num_res_blocks=2, device=device)
state_dict = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded successfully on device: {device}")


def find_best_move(board: Board) -> Move:
    """
    Determine the best move that the model can find given some board.
    """
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Use iterative deepening
    search_depth = 3  # Adjust depth as needed
    maximizing = board.turn  # True if white to move, False if black

    best_move = None
    best_eval = None

    # Iterative deepening
    for depth in range(1, search_depth + 1):
        evaluation, move = alpha_beta(board, depth, maximizingPlayer=maximizing)
        best_move = move
        best_eval = evaluation

    return best_move


def alpha_beta(board: Board, depth: int, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=True) -> tuple[float, Move]:
    """
    Alpha-beta pruning search that returns (evaluation, best_move).
    """
    legal_moves = list(board.generate_legal_moves())

    # Leaf node
    if depth == 0 or not legal_moves:
        fen = board.fen()
        tensor = get_tensor_bytes_from_fen(fen)
        input_tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(device)

        with torch.no_grad():
            evaluation = model(input_tensor).item()

        return (evaluation, None)

    if maximizingPlayer:
        max_eval = float('-inf')
        best_move = None

        for move in legal_moves:
            board.push(move)
            eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff

        return (max_eval, best_move)

    else:
        min_eval = float('inf')
        best_move = None

        for move in legal_moves:
            board.push(move)
            eval_score, _ = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff

        return (min_eval, best_move)
