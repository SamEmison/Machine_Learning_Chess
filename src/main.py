# The program I will run
# Last edited 10/13/2026

from pathlib import Path

import chess
import chess.pgn
import torch

from model import ChessModel
from game import choose_move


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "games.pgn"
MODEL_PATH = ROOT / "checkpoints" / "chess_model.pt"


def main():
    model = ChessModel()

    if MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        )
        print("Loaded your saved model.")
    else:
        print("Starting with an untrained model.")

    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "You"
    game.headers["Black"] = "ChessML"
    node = game

    print("\nEnter moves in UCI format, such as e2e4.")
    print("Type 'resign' to resign.\n")

    resigned = False

    while not board.is_game_over(claim_draw=True):
        print(board)
        print()

        if board.turn == chess.WHITE:
            move_text = input("Your move: ").strip().lower()

            if move_text == "resign":
                resigned = True
                game.headers["Result"] = "0-1"
                break

            try:
                move = chess.Move.from_uci(move_text)

                if move not in board.legal_moves:
                    print("Illegal move. Try again.\n")
                    continue

            except ValueError:
                print("Invalid move format. Example: e2e4\n")
                continue

        else:
            move = choose_move(board, model)
            print(f"ChessML plays: {move.uci()}")

        board.push(move)
        node = node.add_variation(move)

    if not resigned:
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome else "*"
        game.headers["Result"] = result
    else:
        result = "0-1"

    print("\nFinal board:")
    print(board)
    print(f"\nGame result: {result}")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH, "a", encoding="utf-8") as file:
        exporter = chess.pgn.StringExporter(
            headers=True, variations=True, comments=True
        )
        file.write(game.accept(exporter) + "\n\n")

    print(f"Game saved to: {DATA_PATH}")


if __name__ == "__main__":
    main()