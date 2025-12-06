"""
eval.py

This file contains the logic required for determining the best move to be
played in a given position (see find_best_move). This file also initializes
the pytorch model by reading the weights/ directory.
"""

from pathlib import Path
from chess import Move, Board
import torch


from .GoodKnightCommon.fen_to_tensor import get_tensor_bytes_from_fen
from .GoodKnightModel.chess_cnn import create_model

weights_path = Path(__file__).parent / "weights" / "weights.pth"
print(f"Loading model weights from {weights_path}...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create model and load weights
model = create_model(num_filters=32, num_res_blocks=2, device=DEVICE)
state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded successfully on device: {DEVICE}")


def get_adaptive_depth(board: Board) -> int:
    """
    Calculate search depth based on game phase.

    Uses piece count and move count to determine complexity:
    - Opening (many pieces): shallow depth
    - Midgame: medium depth
    - Endgame (few pieces): deep depth

    Always returns odd depth to maintain evaluation consistency.
    """
    # Count total pieces on the board
    piece_count = len(board.piece_map())

    # Count legal moves (branching factor indicator)
    move_count = board.legal_moves.count()

    # Adaptive depth based on piece count
    if piece_count <= 6:  # Late endgame (K+R vs K, etc)
        depth = 7
    elif piece_count <= 10:  # Endgame
        depth = 5
    elif piece_count <= 16:  # Late midgame
        depth = 5
    elif piece_count <= 20:  # Early midgame
        depth = 3
    else:  # Opening
        depth = 3

    # Further adjust based on branching factor
    # If very few moves available, can search deeper
    if move_count < 10 and piece_count <= 10:
        depth += 2  # Add 2 to keep it odd

    print(f"Pieces: {piece_count}, Moves: {move_count}, Depth: {depth}")
    return depth


def find_best_move(board: Board) -> Move:
    """
    Determine the best move that the model can find given some board.
    """
    legal_moves = list(board.generate_legal_moves())
    if not legal_moves:
        raise ValueError("No legal moves available (i probably lost didn't i)")

    print(f"Finding best move for {'White' if board.turn else 'Black'}")

    # Use adaptive depth based on game phase
    depth = get_adaptive_depth(board)
    _, best_move = alpha_beta(board, depth=depth)

    print(f"Found best move of {best_move}")
    return best_move


def alpha_beta(
    board: Board,
    depth: int,
    alpha=float("-inf"),
    beta=float("inf"),
) -> tuple[float, Move]:
    """
    Alpha-beta pruning search that returns (evaluation, best_move).

    The model outputs evaluations relative to the side to move:
    - Positive = good for current player
    - Negative = bad for current player

    We always maximize from the current player's perspective, and negate
    evaluations when passing them up from the opponent's moves.
    """
    legal_moves = list(board.generate_legal_moves())

    # Leaf node - evaluate position from current player's perspective
    if depth == 0 or not legal_moves:
        fen = board.fen()
        tensor = get_tensor_bytes_from_fen(fen)
        input_tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            evaluation = model(input_tensor).item()

        return (evaluation, None)

    # Always maximize from current player's perspective
    max_eval = float("-inf")
    best_move = None

    for move in legal_moves:
        board.push(move)
        # Opponent's evaluation is negated (their good is our bad)
        eval_score, _ = alpha_beta(board, depth - 1, -beta, -alpha)
        eval_score = -eval_score  # Negate opponent's score
        board.pop()
        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move

        alpha = max(alpha, eval_score)
        if beta <= alpha:
            break  # Beta cutoff

    return (max_eval, best_move)
