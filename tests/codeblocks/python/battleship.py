# ...
    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found")

        if len(game.ships) < 5:
            raise ValueError("All ships must be placed before starting turns")

        target_row, target_column = turn.target["row"], turn.target["column"]
        target_column_index = ord(target_column) - ord("A")

        if (target_row, target_column_index) not in game.board:
            game.board[(target_row, target_column_index)] = "miss"
            game.turns.append(turn)
            return TurnResponse(result="miss", ship_type=None)

        ship_type = game.board[(target_row, target_column_index)]
        if ship_type == "miss" or ship_type == "hit":
            game.board[(target_row, target_column_index)] = "hit"
            game.turns.append(turn)
            return TurnResponse(result="hit", ship_type=ship_type)

        game.board[(target_row, target_column_index)] = "hit"
        game.turns.append(turn)

        for row, column in game.board:
            if game.board[(row, column)] == ship_type:
                return TurnResponse(result="hit", ship_type=ship_type)

        return TurnResponse(result="sunk", ship_type=ship_type)
# ...