import random
from typing import List, Dict, Optional, Tuple
from models import AbstractBattleship, ShipPlacement, Turn, TurnResponse, GameStatus, Game

class Battleship(AbstractBattleship):
    def __init__(self):
        self.games: Dict[str, Game] = {}

    def create_game(self) -> str:
        game_id = str(random.randint(1000, 9999))
        self.games[game_id] = Game(
            game_id=game_id,
            players=[],
            board={},
            ships=[],
            turns=[]
        )
        return game_id

    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        game = self.games.get(game_id)
        if not game:
            raise ValueError("Invalid game ID")

        ship_type = placement.ship_type.lower()
        if ship_type not in self.SHIP_LENGTHS:
            raise ValueError("Invalid ship type")

        # Check if the specific type of ship has already been placed
        for ship in game.ships:
            if ship.ship_type == ship_type:
                raise ValueError(f"{ship_type.title()} is already placed. Cannot place more.")

        start_row = placement.start.get("row")
        start_column = placement.start.get("column")
        direction = placement.direction.lower()

        if not (1 <= start_row <= 10):
            raise ValueError("Row must be between 1 and 10 inclusive.")

        if start_column not in list("ABCDEFGHIJ"):
            raise ValueError("Column must be one of A, B, C, D, E, F, G, H, I, J.")

        if direction not in ["horizontal", "vertical"]:
            raise ValueError("Invalid ship direction")

        ship_length = self.SHIP_LENGTHS[ship_type]

        if direction == "horizontal":
            if start_column > chr(ord("J") - ship_length + 1):
                raise ValueError("Ship extends beyond board boundaries")

            for column in range(ord(start_column), ord(start_column) + ship_length):
                if (start_row, column - ord("A")) in game.board:
                    raise ValueError("Ships overlap")

            for column in range(ord(start_column), ord(start_column) + ship_length):
                game.board[(start_row, column - ord("A"))] = "ship"

        else:
            if start_row > 10 - ship_length + 1:
                raise ValueError("Ship extends beyond board boundaries")

            for row in range(start_row, start_row + ship_length):
                if (row, ord(start_column) - ord("A")) in game.board:
                    raise ValueError("Ships overlap")

                game.board[(row, ord(start_column) - ord("A"))] = "ship"

        game.ships.append(placement)

    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        game = self.games.get(game_id)
        if not game:
            raise ValueError("Invalid game ID")

        if not game.ships:
            raise ValueError("All ships must be placed before starting turns")

        target_row = turn.target.get("row")
        target_column = turn.target.get("column")

        if not (1 <= target_row <= 10):
            raise ValueError("Row must be between 1 and 10 inclusive.")

        if target_column not in list("ABCDEFGHIJ"):
            raise ValueError("Column must be one of A, B, C, D, E, F, G, H, I, J.")

        target_key = (target_row, ord(target_column) - ord("A"))

        if target_key in game.board:
            if game.board[target_key] == "ship":
                game.board[target_key] = "hit"
                ship_type = self.get_ship_type(game, target_key)
                if self.is_ship_sunk(game, ship_type):
                    return TurnResponse(result="sunk", ship_type=ship_type)
                return TurnResponse(result="hit", ship_type=ship_type)
            else:
                return TurnResponse(result="miss", ship_type=None)
        else:
            return TurnResponse(result="miss", ship_type=None)

    def get_game_status(self, game_id: str) -> GameStatus:
        game = self.games.get(game_id)
        if not game:
            raise ValueError("Invalid game ID")

        if not game.ships:
            return GameStatus(is_game_over=False, winner=None)

        for ship_placement in game.ships:
            ship_type = ship_placement.ship_type.lower()
            if not self.is_ship_sunk(game, ship_type):
                return GameStatus(is_game_over=False, winner=None)

        return GameStatus(is_game_over=True, winner=game.players[0])

    def get_winner(self, game_id: str) -> str:
        game = self.games.get(game_id)
        if not game:
            raise ValueError("Invalid game ID")

        if not game.ships:
            raise ValueError("Game is not over yet")

        for ship_placement in game.ships:
            ship_type = ship_placement.ship_type.lower()
            if not self.is_ship_sunk(game, ship_type):
                raise ValueError("Game is not over yet")

        return game.players[0]

    def get_game(self, game_id: str) -> Optional[Game]:
        return self.games.get(game_id)

    def delete_game(self, game_id: str) -> None:
        self.games.pop(game_id, None)

    def get_ship_type(self, game: Game, target_key: Tuple[int, int]) -> str:
        for ship_placement in game.ships:
            ship_type = ship_placement.ship_type.lower()
            start_row = ship_placement.start.get("row")
            start_column = ship_placement.start.get("column")
            direction = ship_placement.direction.lower()
            ship_length = self.SHIP_LENGTHS[ship_type]

            if direction == "horizontal":
                if (
                    start_row == target_key[0]
                    and start_column <= chr(target_key[1] + ord("A")) <= chr(ord(start_column) + ship_length - 1)
                ):
                    return ship_type
            else:
                if (
                    start_column == chr(target_key[1] + ord("A"))
                    and start_row <= target_key[0] <= start_row + ship_length - 1
                ):
                    return ship_type

        return ""

    def is_ship_sunk(self, game: Game, ship_type: str) -> bool:
        for ship_placement in game.ships:
            if ship_placement.ship_type.lower() == ship_type:
                start_row = ship_placement.start.get("row")
                start_column = ship_placement.start.get("column")
                direction = ship_placement.direction.lower()
                ship_length = self.SHIP_LENGTHS[ship_type]

                if direction == "horizontal":
                    for column in range(ord(start_column), ord(start_column) + ship_length):
                        if game.board[(start_row, column - ord("A"))] != "hit":
                            return False
                else:
                    for row in range(start_row, start_row + ship_length):
                        if game.board[(row, ord(start_column) - ord("A"))] != "hit":
                            return False

                return True

        return False