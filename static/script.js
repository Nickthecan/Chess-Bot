const board = document.querySelector("#board");
const player = document.querySelector("#player");
const infoDisplay = document.querySelector("#info-display");
const width = 8;
let playerGo = 'White';

const boardPosition = [
    r, n, b, q, k, b, n, r,
    p, p, p, p, p, p, p, p,
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    P, P, P, P, P, P, P, P,
    R, N, B, Q, K, B, N, R
];

const internalBoardPosition = [
    'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r',
    'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '',
    'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
    'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'
]

function createBoard() {
    boardPosition.forEach((piece, i) => {
        const square = document.createElement('div');
        square.classList.add('square');
        square.innerHTML = piece;
        square.firstChild && square.firstChild.setAttribute('draggable', true);
        square.setAttribute('square-id', i);
        const row = Math.floor((63 - i) / 8) / 1
        if (row % 2 === 0) {
            square.classList.add(i % 2 === 0 ? "purple" : "white");
        }
        else {
            square.classList.add(i % 2 === 0 ? "white" : "purple");
        }
        board.append(square);
    });
}
// rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
function boardToFEN(board) {
    console.log(board)
    let fen = '';
    let emptyCount = 0;

    for (let rank = 0; rank < 8; rank++) {
        if (rank !== 0)
            fen += '/';
        for (let file = 0; file < 8; file++) {
            const piece = board[rank][file];
            if (piece === '') {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    fen += emptyCount.toString();
                    emptyCount = 0;
                }
                fen += piece;
            }
        }
        if (emptyCount > 0) {
            fen += emptyCount.toString();
            emptyCount = 0;
        }
    }
    if (emptyCount > 0) {
        fen += emptyCount.toString();
    }
    console.log(fen)
    return fen;
}

function fenToBoard(fen) {
    const board = [];
    const ranks = fen.split(' ')[0].split('/');
    ranks.forEach((rank) => {
        const row = [];
        let colIndex = 0;
        for (let i = 0; i < rank.length; i++) {
            const char = rank.charAt(i);
            if (!isNaN(char)) {
                // If the character is a number, it represents an empty square
                const emptySquares = parseInt(char);
                for (let j = 0; j < emptySquares; j++) {
                    row.push('');
                    colIndex++;
                }
            } 
            else {
                // If the character is a piece, add it to the row
                row.push(char);
                colIndex++;
            }
        }
        board.push(row);
    });
    return board;
}

function fetchBoardPosition() {
    fetch("/get_board_position")
    .then(response => response.json())
    .then(data => {
        const updatedBoardPosition = fenToBoard(data.board_position);
        updateBoard(updatedBoardPosition);
    })
    .catch(error => {
        console.error("Error fetching board position:", error);
    });
}

function updateBoard(updatedPieces) {
    const squares = document.querySelectorAll('.square');
    squares.forEach((square, i) => {
        const piece = updatedPieces[Math.floor(i / 8)][i % 8]; 
        square.innerHTML = piece; 
        square.firstChild && square.firstChild.setAttribute('draggable', false);
    });
}

function submitMove() {
    const move = moveInput.value.trim();
    console.log(move)
    if (move !== "") {
        const fen = boardToFEN(internalBoardPosition)
        console.log(fen)
        fetch("/submit_move", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ fen: fen, move: move })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                fetchBoardPosition();
                moveInput.value = "";
            } else {
                console.error("Failed to submit move");
            }
        })
        .catch(error => {
            console.error("Error submitting move:", error);
        });
    }
}

createBoard();


