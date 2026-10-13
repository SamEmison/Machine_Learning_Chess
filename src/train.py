# Training and Saving Model
# Last edited 10/13/2026

from pathlib import Path

import chess
import chess.pgn
import torch
import torch.nn as nn

from model import ChessModel, encode_board


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "games.pgn"
MODEL_PATH = ROOT / "checkpoints" / "chess_model.pt"


def train_model():
    if not DATA_PATH.exists():
        print("No saved games found. Play a game first.")
        return

    positions = []
    targets = []

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        while game := chess.pgn.read_game(file):
            result = game.headers.get("Result", "*")

            if result == "1-0":
                target = 1.0
            elif result == "0-1":
                target = -1.0
            elif result == "1/2-1/2":
                target = 0.0
            else:
                continue

            board = game.board()

            for move in game.mainline_moves():
                positions.append(encode_board(board))
                targets.append(target)
                board.push(move)

            positions.append(encode_board(board))
            targets.append(target)

    if not positions:
        print("No completed games with valid results found.")
        return

    model = ChessModel()

    if MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    x = torch.stack(positions)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    model.train()

    for epoch in range(10):
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_function(predictions, y)
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch + 1}/10 - "
            f"Loss: {loss.item():.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training positions used: {len(positions)}")


if __name__ == "__main__":
    train_model()