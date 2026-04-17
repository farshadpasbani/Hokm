# game_constants.py

# Define suits and ranks for a standard 52-card deck
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
rank_values = {rank: i for i, rank in enumerate(ranks, start=2)}  # 2: 2, ..., Ace: 14

# Fixed observation / action sizing for the DQN (52-card encoding)
STATE_DIM = 114  # see EnhancedPlayer.get_state()
ACTION_DIM = 52


def card_to_index(card):
    """Map a card to a stable0..51 index (suit-major, rank-minor)."""
    return suits.index(card.suit) * len(ranks) + ranks.index(card.rank)


def index_to_card(index):
    """Inverse of card_to_index."""
    if index < 0 or index >= 52:
        raise ValueError(f"Card index out of range: {index}")
    suit = suits[index // len(ranks)]
    rank = ranks[index % len(ranks)]
    return Card(suit, rank)


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


if __name__ == "__main__":
    a = Card("Hearts", "Ace")
    print("a=", a)
    print("a.str()=", a.__str__())
    print("a.__hash__()=", a.__hash__())
    print("a.rank=", a.rank)
    print("a.suit=", a.suit)
    print("a.value=", a.value)
