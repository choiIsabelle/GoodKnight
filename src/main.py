import os
import json
import random
import time
import torch
import numpy as np
import os
from src.utils import chess_manager
from .import_model import load_pytorch_weights
from .getTensorFromFen import get_tensor_bytes_from_fen

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etc
weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'weights.pth')
print(f"Loading model weights from {weights_path}...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_pytorch_weights(weights_path, num_filters=256, num_res_blocks=20, device=device)
print(f"Model loaded successfully on device: {device}")


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    # Evaluate each legal move
    move_evaluations = {}
    for move in legal_moves:
        # Make the move on a copy of the board
        ctx.board.push(move)

        # Get FEN of the resulting position
        fen = ctx.board.fen()

        # Convert FEN to tensor
        tensor_bytes = get_tensor_bytes_from_fen(fen)
        tensor = np.frombuffer(tensor_bytes, dtype=np.uint8).reshape(18, 8, 8)

        # Get model evaluation
        input_tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
        with torch.no_grad():
            evaluation = model(input_tensor).item()

        # Undo the move
        ctx.board.pop()

        # Store evaluation (negate if we're black, since model evaluates from white's perspective)
        if ctx.board.turn:  # White to move
            move_evaluations[move] = evaluation
        else:  # Black to move
            move_evaluations[move] = -evaluation

    # Find the best move (highest evaluation)
    best_move = max(move_evaluations, key=move_evaluations.get)

    # Convert evaluations to probabilities using softmax with temperature
    temperature = 0.1  # Lower temperature = more greedy
    eval_values = np.array(list(move_evaluations.values()))
    exp_values = np.exp((eval_values - np.max(eval_values)) / temperature)
    probabilities = exp_values / np.sum(exp_values)

    move_probs = {move: prob for move, prob in zip(move_evaluations.keys(), probabilities)}
    ctx.logProbabilities(move_probs)
    print(f"Best move evaluation: {move_evaluations[best_move]:.4f}")
    return best_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
