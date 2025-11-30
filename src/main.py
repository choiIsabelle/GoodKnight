from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import torch
import numpy as np
import os
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).parent / 'chess-hacks-training-main'))
sys.path.insert(0, str(Path(__file__).parent / 'GoodKnightCommon'))

from chess_cnn import create_model
from fen_to_tensor import get_tensor_bytes_from_fen

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etc
weights_path = Path(__file__).parent / 'weights' / 'weights.pth'
print(f"Loading model weights from {weights_path}...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model and load weights
model = create_model(num_filters=32, num_res_blocks=2, device=device)
state_dict = torch.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded successfully on device: {device}")


# Global PV table to store principal variations indexed by depth from root
pv_table = {}

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    start = time.time()
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    # Use iterative deepening with PV move ordering
    search_depth = 3  # Adjust depth as needed
    maximizing = ctx.board.turn  # True if white to move, False if black

    # Clear PV table for new search
    global pv_table
    pv_table = {}

    best_move = None
    best_eval = None

    # Iterative deepening to build up PV for move ordering
    for depth in range(1, search_depth + 1):
        evaluation, move, pv = alpha_beta(ctx, depth, maximizingPlayer=maximizing)
        best_move = move
        best_eval = evaluation

        # Store PV in table indexed by depth from root
        pv_table = {}
        for i, pv_move in enumerate(pv):
            pv_table[i] = pv_move

    end = time.time()

    print(f"Found best move evaluation of {best_eval:.4f} in {round(end - start, 3)}s")

    return best_move


def alpha_beta(ctx: GameContext, depth: int, alpha=float('-inf'), beta=float('inf'), maximizingPlayer=True, ply_from_root=0):
    """
    Alpha-beta pruning search with PV move ordering that returns (evaluation, best_move, pv).
    """
    legal_moves = list(ctx.board.generate_legal_moves())

    # Leaf node
    if depth == 0 or not legal_moves:
        fen = ctx.board.fen()
        tensor = get_tensor_bytes_from_fen(fen)
        input_tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(device)

        with torch.no_grad():
            evaluation = model(input_tensor).item()

        return (evaluation, None, [])

    # PV move ordering: search PV move first if available
    pv_move = pv_table.get(ply_from_root)
    if pv_move and pv_move in legal_moves:
        # Move PV move to front of list
        legal_moves.remove(pv_move)
        legal_moves.insert(0, pv_move)

    if maximizingPlayer:
        max_eval = float('-inf')
        best_move = None
        best_pv = []

        for move in legal_moves:
            ctx.board.push(move)
            eval_score, _, child_pv = alpha_beta(ctx, depth - 1, alpha, beta, False, ply_from_root + 1)
            ctx.board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                best_pv = [move] + child_pv

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff

        return (max_eval, best_move, best_pv)

    else:
        min_eval = float('inf')
        best_move = None
        best_pv = []

        for move in legal_moves:
            ctx.board.push(move)
            eval_score, _, child_pv = alpha_beta(ctx, depth - 1, alpha, beta, True, ply_from_root + 1)
            ctx.board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                best_pv = [move] + child_pv

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff

        return (min_eval, best_move, best_pv)

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
