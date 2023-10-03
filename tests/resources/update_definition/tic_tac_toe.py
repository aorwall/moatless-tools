def print_board(board):
    for row in board:
        print(' '.join(row))

def check_win(board):
    for row in board:
        if row.count(row[0]) == len(row) and row[0] != '0':
            return True

    for col in range(len(board)):
        check = []
        for row in board:
            check.append(row[col])
        if check.count(check[0]) == len(check) and check[0] != '0':
            return True

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '0':
        return True
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '0':
        return True

    return False

def check_draw(board):
    for row in board:
        if '0' in row:
            return False
    return True

def tic_tac_toe():
    board = [['0', '0', '0'] for _ in range(3)]
    player = 1

    while True:
        print_board(board)
        print(f"Player {player}'s turn. Enter move (row,col): ", end="")
        move = input().strip().split(',')
        row, col = int(move[0]), int(move[1])

        if board[row][col] != '0':
            print("Invalid move. Try again.")
            continue

        board[row][col] = str(player)

        if check_win(board):
            print(f"Player {player} won!")
            break

        if check_draw(board):
            print("Draw")
            break

        player = 1 if player == 2 else 2

if __name__ == "__main__":
    tic_tac_toe()

