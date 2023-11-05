from abstract_class import AbstractBattleship, Game, GameStatus, ShipPlacement, Turn, TurnResponse
from typing import Dict, List, Optional, Tuple
from uuid import uuid4


class Battleship(AbstractBattleship):
    def __init__(self):
        self.games: Dict[str, Game] = {}

    def create_game(self) -> str:
        game_id = str(uuid4())
        self.games[game_id] = Game(game_id=game_id, players=["Player1", "Player2"], board={}, ships=[], turns=[])
        return game_id

    def delete_game(self, game_id: str) -> None:
        del self.games[game_id]

    def get_game(self, game_id: str) -> Game:
        return self.games[game_id]

    def create_ship_placement(self, game_id: str, placement: ShipPlacement) -> None:
        game = self.get_game(game_id)
        ship_length = self.SHIP_LENGTHS[placement.ship_type]

        start_row, start_column = placement.start["row"], ord(placement.start["column"]) - ord("A")
        direction = placement.direction

        if direction == "horizontal":
            end_row, end_column = start_row, start_column + ship_length
        elif direction == "vertical":
            end_row, end_column = start_row + ship_length, start_column
        else:
            raise ValueError("Invalid ship direction")

        if not (1 <= end_row <= 12) or not (0 <= end_column < 12):
            raise ValueError("Ship extends beyond board boundaries")

        for row in range(start_row, end_row + 1):
            for column in range(start_column, end_column + 1):
                if (row, column) in game.board:
                    raise ValueError("Ships cannot overlap")

                game.board[(row, column)] = {"ship_type": placement.ship_type, "hit": False}

        game.ships.append(placement)

    def create_turn(self, game_id: str, turn: Turn) -> TurnResponse:
        game = self.get_game(game_id)

        if len(game.ships) < 6:
            raise ValueError("All ships must be placed before starting turns")

        target_row, target_column = turn.target["row"], ord(turn.target["column"]) - ord("A")
        target_key = (target_row, target_column)

        if target_key not in game.board:
            game.board[target_key] = {"hit": False}
            game.turns.append(turn)
            return TurnResponse(result="miss", ship_type=None)

        if game.board[target_key]["hit"]:
            raise ValueError("Cannot target a grid cell that has already been hit")

        game.board[target_key]["hit"] = True
        game.turns.append(turn)

        for ship_placement in game.ships:
            start_row, start_column = ship_placement.start["row"], ord(ship_placement.start["column"]) - ord("A")
            ship_length = self.SHIP_LENGTHS[ship_placement.ship_type]

            if ship_placement.direction == "horizontal":
                end_row, end_column = start_row, start_column + ship_length
            else:
                end_row, end_column = start_row + ship_length, start_column

            if all(game.board[(row, column)]["hit"] for row in range(start_row, end_row + 1) for column in range(start_column, end_column + 1)):
                return TurnResponse(result="sunk", ship_type=ship_placement.ship_type)

        return TurnResponse(result="hit", ship_type=game.board[target_key]["ship_type"])

    def get_game_status(self, game_id: str) -> GameStatus:
        game = self.get_game(game_id)

        if len(game.turns) < 2 * 6:  # Each player must place all their ships before the game can end
            return GameStatus(is_game_over=False, winner=None)

        player1_turns = game.turns[::2]
        player2_turns = game.turns[1::2]

        player1_sunk_ships = sum(1 for turn in player1_turns if self.create_turn(game_id, turn).result == "sunk")
        player2_sunk_ships = sum(1 for turn in player2_turns if self.create_turn(game_id, turn).result == "sunk")

        if player1_sunk_ships == 6:
            return GameStatus(is_game_over=True, winner="Player1")
        elif player2_sunk_ships == 6:
            return GameStatus(is_game_over=True, winner="Player2")
        else:
            return GameStatus(is_game_over=False, winner=None)

    def get_winner(self, game_id: str) -> str:
        return self.get_game_status(game_id).winner
