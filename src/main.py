import os
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess as _chess
from chess import Move

from .utils import chess_manager, GameContext

# Load move_to_idx mapping and model.pt from src/weights
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
MOVE_TO_IDX = {}
IDX_TO_MOVE = {}
try:
    with open(os.path.join(WEIGHTS_DIR, "model.pt"), "rb") as _:
        pass  # Just to check existence
    with open(os.path.join(WEIGHTS_DIR, "move_to_idx.json"), "r") as f:
        MOVE_TO_IDX = json.load(f)
        IDX_TO_MOVE = {int(v): k for k, v in MOVE_TO_IDX.items()}
except Exception:
    MOVE_TO_IDX = {}
    IDX_TO_MOVE = {}

# Load your custom model from src/weights/best_chess_model.pth
MODEL_PATH = os.path.join(WEIGHTS_DIR, "best_chess_model.pth")



class ResidualBlock(nn.Module):
    def __init__(self, num_filters, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out



class ChessEvaluationCNN(nn.Module):
    def __init__(self, num_filters=128, num_res_blocks=6, dropout_rate=0.3):
        super(ChessEvaluationCNN, self).__init__()
        self.conv_input = nn.Conv2d(18, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, dropout_rate=dropout_rate) for _ in range(num_res_blocks)
        ])
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_dropout = nn.Dropout(dropout_rate)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        # Policy head (not used for evaluation, but needed for loading weights)
        _ = F.relu(self.policy_bn(self.policy_conv(x)))
        # Value head
        val = F.relu(self.value_bn(self.value_conv(x)))
        val = val.view(val.size(0), -1)
        val = F.relu(self.value_fc1(val))
        val = self.value_dropout(val)
        val = self.value_fc2(val)
        return val


def fen_to_tensor_custom(fen: str) -> np.ndarray:
    piece_to_index = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    fen_parts = fen.split(' ')
    board_part = fen_parts[0]
    active_player = fen_parts[1] if len(fen_parts) > 1 else 'w'
    castling_part = fen_parts[2] if len(fen_parts) > 2 else '-'
    enpassant_part = fen_parts[3] if len(fen_parts) > 3 else '-'
    rows = board_part.split('/')
    for r, row in enumerate(rows):
        c = 0
        for char in row:
            if char.isdigit():
                c += int(char)
            else:
                if char in piece_to_index:
                    tensor[piece_to_index[char], r, c] = 1
                c += 1
    if castling_part != '-':
        if 'K' in castling_part:
            tensor[12, :, :] = 1
        if 'Q' in castling_part:
            tensor[13, :, :] = 1
        if 'k' in castling_part:
            tensor[14, :, :] = 1
        if 'q' in castling_part:
            tensor[15, :, :] = 1
    if enpassant_part != '-':
        col = ord(enpassant_part[0]) - ord('a')
        row = 8 - int(enpassant_part[1])
        tensor[16, row, col] = 1
    if active_player == 'w':
        tensor[17, :, :] = 1
    return tensor


_MODEL = None
_DEVICE = torch.device("cpu")



def _load_custom_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        _MODEL = ChessEvaluationCNN(num_filters=128, num_res_blocks=6, dropout_rate=0.3)
        if os.path.exists(MODEL_PATH):
            _MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=_DEVICE))
        _MODEL.eval()
        return _MODEL
    except Exception as e:
        print(f"Error loading model: {e}, model path was {MODEL_PATH}")
        _MODEL = None
        return None


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Use your trained model to evaluate all legal moves and pick the best one.
    """
    print("GoodKnight: computing move via custom model")
    print(ctx.board.move_stack)

    # raise NotImplementedError("Custom model evaluation not implemented")

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    model = _load_custom_model()
    if True:
        # fallback to random
        move_weights = [random.random() for _ in legal_moves]
        total_weight = sum(move_weights)
        move_probs = {m: w / total_weight for m, w in zip(legal_moves, move_weights)}
        ctx.logProbabilities(move_probs)
        return random.choices(legal_moves, weights=move_weights, k=1)[0]

    # Evaluate each legal move using your model
    fen = ctx.board.fen()
    move_scores = []
    for move in legal_moves:
        board_copy = ctx.board.copy()
        board_copy.push(move)
        move_fen = board_copy.fen()
        tensor = fen_to_tensor_custom(move_fen)
        tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = model(tensor).item()
        move_scores.append(score)

    # Pick the move with the highest score
    best_idx = int(np.argmax(move_scores))
    best_move = legal_moves[best_idx]

    # Normalize scores for logProbabilities
    scores = np.array(move_scores)
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()
    move_probs = {m: float(p) for m, p in zip(legal_moves, probs)}
    ctx.logProbabilities(move_probs)
    return best_move


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etc


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
