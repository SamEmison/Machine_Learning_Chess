# Model File 
# Last edited 10/13/2026


import torch
import torch.nn as nn
import chess


class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(769, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)


def encode_board(board):
    position = [0.0] * 769

    for square, piece in board.piece_map().items():
        piece_index = piece.piece_type - 1

        if piece.color == chess.BLACK:
            piece_index += 6

        position[piece_index * 64 + square] = 1.0

    position[768] = 1.0 if board.turn == chess.WHITE else 0.0

    return torch.tensor(position, dtype=torch.float32)