from typing import List
from models import AbstractBattleship, ShipPlacement, Turn, TurnResponse, GameStatus, Game

class Battleship(AbstractBattleship):
    def __init__(self):
        self.games = {}
