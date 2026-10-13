# Chess gameplay and move selection
# Last edited 10/13/2026

import chess
import torch

from model import ChessModel, encode_board


def choose_move(board, model):
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    best_move = None
    best_score = float("-inf") if board.turn == chess.WHITE else float("inf")

    model.eval()

    with torch.no_grad():
        for move in legal_moves:
            board.push(move)
            position = encode_board(board)
            score = model(position).item()
            board.pop()

            if board.turn == chess.WHITE and score > best_score:
                best_score = score
                best_move = move
            elif board.turn == chess.BLACK and score < best_score:
                best_score = score
                best_move = move

    return best_move