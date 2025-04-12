# Define suits and ranks
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
rank_values = {rank: i for i, rank in enumerate(ranks, start=2)}


class Card:
    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
        self.value = rank_values[rank]

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return str(self)

    @classmethod
    def from_string(cls, card_string):
        """Parse a card string into a Card object.
        Example: "Ace of Hearts" -> Card("Hearts", "Ace")
        """
        try:
            # Split on " of " to handle ranks with multiple words
            parts = card_string.split(" of ")
            if len(parts) != 2:
                raise ValueError(f"Invalid card string format: {card_string}")

            rank = parts[0]
            suit = parts[1]

            # Validate rank and suit
            if rank not in ranks:
                raise ValueError(f"Invalid rank: {rank}")
            if suit not in suits:
                raise ValueError(f"Invalid suit: {suit}")

            return cls(suit, rank)
        except Exception as e:
            raise ValueError(
                f"Invalid card string format: {card_string}. Error: {str(e)}"
            )

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
