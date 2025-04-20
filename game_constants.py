# game_constants.py

# Define suits and ranks for a standard 52-card deck
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
rank_values = {rank: i for i, rank in enumerate(ranks, start=2)}  # 2: 2, ..., Ace: 14


class Card:
    def __init__(self, suit, rank):
        if suit not in suits:
            raise ValueError(f"Invalid suit: {suit}")
        if rank not in ranks:
            raise ValueError(f"Invalid rank: {rank}")
        self.suit = suit
        self.rank = rank
        self.value = rank_values[rank]

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    @classmethod
    def from_string(cls, card_string):
        """Parse a card string into a Card object (e.g., 'Ace of Hearts')."""
        try:
            parts = card_string.split(" of ")
            if len(parts) != 2:
                raise ValueError(f"Invalid card string format: {card_string}")
            rank, suit = parts[0], parts[1]
            if rank not in ranks:
                raise ValueError(f"Invalid rank: {rank}")
            if suit not in suits:
                raise ValueError(f"Invalid suit: {suit}")
            return cls(suit, rank)
        except Exception as e:
            raise ValueError(
                f"Failed to parse card string: {card_string}. Error: {str(e)}"
            )
