# Gui interface using streamlit


import sys
from pathlib import Path

import chess
import chess.pgn
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import ChessModel
from game import choose_move
from train import train_model

DATA_PATH = ROOT / "data" / "games.pgn"
MODEL_PATH = ROOT / "checkpoints" / "chess_model.pt"

st.set_page_config(
    page_title="ChessML",
    page_icon="♟",
    layout="wide",
)

# ----------------------------
# Styling
# ----------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #171a21;
        color: #e8e8e8;
    }

    h1, h2, h3 {
        color: #f0f0f0;
    }

    div[data-testid="stButton"] button {
        transition: filter 0.12s ease;
    }

    .st-key-chessboard div[data-testid="stButton"] button {
        height: clamp(38px, 7vw, 76px);
        min-height: clamp(38px, 7vw, 76px);
        padding: 0;
        border-radius: 0;
        font-size: clamp(20px, 4vw, 43px);
        font-weight: 500;
        box-shadow: none;
        border: none;
    }

    .st-key-chessboard div[data-testid="stButton"] button:hover {
        filter: brightness(1.12);
        outline: 3px solid #f6f669;
        outline-offset: -3px;
    }

    .st-key-chessboard div[data-testid="column"] {
        padding: 0 !important;
        min-width: 0 !important;
    }

    .st-key-chessboard div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
    }

    .st-key-chessboard button[kind="primary"] {
        background-color: #769656;
        color: white;
    }

    .st-key-chessboard button[kind="secondary"] {
        background-color: #eeeed2;
        color: #222222;
    }

    div[data-testid="stMetric"] {
        background-color: #222831;
        padding: 12px;
        border-radius: 8px;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Model
# ----------------------------

@st.cache_resource
def load_model():
    model = ChessModel()

    if MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True,
            )
        )

    model.eval()
    return model


# ----------------------------
# Game state
# ----------------------------

def new_game():
    st.session_state.board = chess.Board()
    st.session_state.game = chess.pgn.Game()
    st.session_state.node = st.session_state.game
    st.session_state.selected = None
    st.session_state.saved = False
    st.session_state.message = "Your turn — you play White."


def save_game(result):
    if st.session_state.saved:
        return

    game = st.session_state.game
    game.headers["White"] = "You"
    game.headers["Black"] = "ChessML"
    game.headers["Result"] = result

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    exporter = chess.pgn.StringExporter(
        headers=True,
        variations=True,
        comments=True,
    )

    with open(DATA_PATH, "a", encoding="utf-8") as file:
        file.write(game.accept(exporter) + "\n\n")

    st.session_state.saved = True


def finish_game():
    board = st.session_state.board
    outcome = board.outcome(claim_draw=True)

    result = outcome.result() if outcome else "*"
    save_game(result)

    if result == "1-0":
        st.session_state.message = "You won the game!"
    elif result == "0-1":
        st.session_state.message = "ChessML won the game!"
    else:
        st.session_state.message = "The game ended in a draw."


# ----------------------------
# Move handling
# ----------------------------

def play_move(move):
    board = st.session_state.board

    board.push(move)
    st.session_state.node = st.session_state.node.add_variation(move)

    if board.is_game_over(claim_draw=True):
        finish_game()
        return

    # Let the model choose its response.
    with st.spinner("ChessML is thinking..."):
        ai_move = choose_move(board, load_model())

    if ai_move is None:
        finish_game()
        return

    board.push(ai_move)
    st.session_state.node = st.session_state.node.add_variation(ai_move)

    st.session_state.message = (
        f"ChessML played {ai_move.uci()}. Your turn."
    )

    if board.is_game_over(claim_draw=True):
        finish_game()


def handle_square(square):
    board = st.session_state.board

    if board.turn != chess.WHITE or board.is_game_over():
        return

    selected = st.session_state.selected

    if selected is None:
        piece = board.piece_at(square)

        if piece and piece.color == chess.WHITE:
            st.session_state.selected = square
        return

    if selected == square:
        st.session_state.selected = None
        return

    candidates = [
        move
        for move in board.legal_moves
        if move.from_square == selected
        and move.to_square == square
    ]

    if candidates:
        # Automatically promote pawns to queens.
        move = next(
            (
                candidate
                for candidate in candidates
                if candidate.promotion == chess.QUEEN
            ),
            candidates[0],
        )

        st.session_state.selected = None
        play_move(move)
        return

    # Clicking another white piece selects that piece.
    piece = board.piece_at(square)

    if piece and piece.color == chess.WHITE:
        st.session_state.selected = square
    else:
        st.session_state.selected = None


# ----------------------------
# Initialize
# ----------------------------

if "board" not in st.session_state:
    new_game()

board = st.session_state.board


# ----------------------------
# Page layout
# ----------------------------

st.title("♟ ChessML")
st.caption("A machine-learning chess project")

left, right = st.columns([3, 2], gap="large")


# ----------------------------
# Chessboard
# ----------------------------

with left:
    st.subheader("Chessboard")
    st.write(st.session_state.message)

    with st.container(key="chessboard"):
        for rank in range(7, -1, -1):
            columns = st.columns(8, gap=0)

            for file_index, column in enumerate(columns):
                square = chess.square(file_index, rank)
                piece = board.piece_at(square)

                symbol = piece.unicode_symbol() if piece else "\u00a0"

                is_light = (file_index + rank) % 2 == 1
                square_color = "secondary" if is_light else "primary"

                # Mark the currently selected square.
                if st.session_state.selected == square:
                    symbol = f"● {symbol}"

                with column:
                    st.button(
                        symbol,
                        key=f"square_{square}",
                        type=square_color,
                        use_container_width=True,
                        on_click=handle_square,
                        args=(square,),
                        disabled=board.is_game_over(),
                    )

    st.caption(
        "Click one of your white pieces, then click its destination."
    )

    # Game controls.
    control1, control2 = st.columns(2)

    with control1:
        if st.button("New Game", use_container_width=True):
            new_game()
            st.rerun()

    with control2:
        if st.button(
            "Resign",
            disabled=board.is_game_over(),
            use_container_width=True,
        ):
            save_game("0-1")
            st.session_state.message = "You resigned. ChessML wins."
            st.rerun()


# ----------------------------
# Information sidebar
# ----------------------------

with right:
    st.subheader("Game Information")

    st.metric(
        "Current turn",
        "Your turn" if board.turn == chess.WHITE else "ChessML",
    )

    st.metric("Full moves played", board.fullmove_number - 1)

    if board.is_check():
        st.warning("Check!")

    if board.is_checkmate():
        st.error("Checkmate!")
    elif board.is_stalemate():
        st.info("Stalemate.")
    elif board.is_game_over(claim_draw=True):
        st.info("Game over.")

    st.divider()

    st.subheader("Move History")

    moves = list(st.session_state.game.mainline_moves())
    history = []

    for index in range(0, len(moves), 2):
        white_move = moves[index].uci()
        black_move = (
            moves[index + 1].uci()
            if index + 1 < len(moves)
            else ""
        )

        history.append(
            f"{index // 2 + 1}. {white_move}   {black_move}"
        )

    st.code(
        "\n".join(history) if history else "No moves yet",
        language=None,
    )

    st.divider()

    st.subheader("Model & Training")

    st.write(
        "**Model status:**",
        "Saved model found" if MODEL_PATH.exists()
        else "Using untrained model",
    )

    if st.button("Train on Saved Games", use_container_width=True):
        with st.spinner("Training on completed games..."):
            train_model()
            load_model.clear()

        st.success("Training finished.")
        st.rerun()

    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as file:
            saved_games = sum(
                1
                for line in file
                if line.startswith("[Result ")
            )
    else:
        saved_games = 0

    st.metric("Saved games", saved_games)

    st.caption(
        "Completed games are saved locally in data/games.pgn."
    )
