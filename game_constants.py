# game_constants.py

# Define suits and ranks for a standard 52-card deck
suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
rank_values = {rank: i for i, rank in enumerate(ranks, start=2)}  # 2: 2, ..., Ace: 14

# -----------------------------------------------------------------------------
# Observation / action sizing for the NFSP networks.
# -----------------------------------------------------------------------------
#
# STATE_DIM is the flat feature vector produced by EnhancedPlayer.get_state().
# It's been designed to make the "unwritten lemmas" that competent human
# Hokm players rely on *actually learnable*: card-level memory for promotion,
# per-opponent voids from failure-to-follow, trick-position awareness, who
# is currently winning the trick, and Hakem-awareness.
#
# Layout (indices are half-open ranges; see get_state() for the canonical
# source of truth):
#
#     [   0:  52)  my hand, one-hot per card                          (52)
#     [  52: 104)  cards already played this hand, one-hot            (52)
#     [ 104: 116)  void flags per other player × suit (3×4)           (12)
#                   order: RHO, partner, LHO (seat-relative)
#     [ 116: 168)  cards in the current in-progress trick, one-hot    (52)
#     [ 168: 172)  lead suit of current trick, one-hot                ( 4)
#     [ 172: 176)  trick position (1st/2nd/3rd/4th to play)           ( 4)
#     [ 176: 181)  current winner seat (empty/me/partner/LHO/RHO)     ( 5)
#     [ 181: 182)  current winning card value, normalised (/14)       ( 1)
#     [ 182: 183)  Hakem-is-me flag                                   ( 1)
#     [ 183: 184)  Hakem-is-my-partner flag                           ( 1)
#     [ 184: 185)  my team's trick count, normalised (/7)             ( 1)
#     [ 185: 186)  opp team's trick count, normalised (/7)            ( 1)
#     [ 186: 190)  trump suit, one-hot                                ( 4)
#     [ 190: 194)  my hand per-suit count, normalised (/13)           ( 4)
#
# STATE_DIM = 194.
#
# Any checkpoint with a different input dim (e.g. legacy 114-dim runs) will
# silently have its input-layer weights dropped by `_load_state_dict_compat`
# — the deeper layers still load, but the first layer starts from fresh init
# and needs a short retrain.
STATE_DIM = 194
ACTION_DIM = 52

# Named offsets for external consumers / tests that want to read slices.
STATE_LAYOUT = {
    "hand": (0, 52),
    "cards_played": (52, 104),
    "voids": (104, 116),
    "current_trick": (116, 168),
    "lead_suit": (168, 172),
    "trick_position": (172, 176),
    "winner_seat": (176, 181),
    "winner_value": (181, 182),
    "hakem_is_me": (182, 183),
    "hakem_is_partner": (183, 184),
    "team_tricks": (184, 185),
    "opp_tricks": (185, 186),
    "trump": (186, 190),
    "hand_suit_counts": (190, 194),
}


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
