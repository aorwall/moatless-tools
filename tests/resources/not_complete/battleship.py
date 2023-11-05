
# ... code

class Battleship(AbstractBattleship):
    # ... code

    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found")

        if placement.ship_type not in self.SHIP_LENGTHS:
            raise ValueError("Invalid ship type")

        # ... rest of the code

    # ... code

    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        game = self.get_game(game_id)
        if not game:
            raise ValueError("Game not found")

        if len(game.ships) < 5:
            raise ValueError("All ships must be placed before starting turns")

        target_row, target_column = turn.target["row"], turn.target["column"]
        target_column_index = ord(target_column) - ord("A")

        if (target_row, target_column_index) in game.board and game.board[(target_row, target_column_index)] in ["miss", "hit"]:
            raise ValueError("Cannot target the same cell twice")

        # ... rest of the code

    # ... code