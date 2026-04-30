#include "card.h"
#include "game_constants.h"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace hokm {

Card::Card() : suit(""), rank(""), value(0) {}

Card::Card(const std::string& suit, const std::string& rank) {
    if (std::find(SUITS.begin(), SUITS.end(), suit) == SUITS.end()) {
        throw std::invalid_argument("Invalid suit: " + suit);
    }
    if (std::find(RANKS.begin(), RANKS.end(), rank) == RANKS.end()) {
        throw std::invalid_argument("Invalid rank: " + rank);
    }
    this->suit = suit;
    this->rank = rank;
    this->value = get_rank_value(rank);
}

std::string Card::to_string() const {
    return rank + " of " + suit;
}

bool Card::operator==(const Card& other) const {
    return suit == other.suit && rank == other.rank;
}

bool Card::operator!=(const Card& other) const {
    return !(*this == other);
}

Card Card::from_string(const std::string& card_string) {
    size_t pos = card_string.find(" of ");
    if (pos == std::string::npos) {
        throw std::invalid_argument("Invalid card string format: " + card_string);
    }
    std::string rank = card_string.substr(0, pos);
    std::string suit = card_string.substr(pos + 4);
    return Card(suit, rank);
}

int card_to_index(const Card& card) {
    auto suit_it = std::find(SUITS.begin(), SUITS.end(), card.suit);
    auto rank_it = std::find(RANKS.begin(), RANKS.end(), card.rank);
    
    if (suit_it == SUITS.end() || rank_it == RANKS.end()) {
        throw std::invalid_argument("Invalid card for indexing");
    }
    
    int suit_idx = std::distance(SUITS.begin(), suit_it);
    int rank_idx = std::distance(RANKS.begin(), rank_it);
    
    return suit_idx * RANKS.size() + rank_idx;
}

Card index_to_card(int index) {
    if (index < 0 || index >= 52) {
        throw std::invalid_argument("Card index out of range: " + std::to_string(index));
    }
    int suit_idx = index / RANKS.size();
    int rank_idx = index % RANKS.size();
    return Card(SUITS[suit_idx], RANKS[rank_idx]);
}

} // namespace hokm
