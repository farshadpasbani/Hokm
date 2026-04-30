#include "deck.h"
#include "game_constants.h"
#include <algorithm>
#include <stdexcept>
#include <chrono>

namespace hokm {

Deck::Deck() {
    for (const auto& suit : SUITS) {
        for (const auto& rank : RANKS) {
            cards.push_back(Card(suit, rank));
        }
    }
}

void Deck::shuffle(std::mt19937* rng) {
    if (rng) {
        std::shuffle(cards.begin(), cards.end(), *rng);
    } else {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 default_rng(seed);
        std::shuffle(cards.begin(), cards.end(), default_rng);
    }
}

std::vector<Card> Deck::deal(int num_cards) {
    if (cards.size() < static_cast<size_t>(num_cards)) {
        throw std::invalid_argument("Not enough cards in deck to deal " + std::to_string(num_cards) + " cards");
    }
    
    std::vector<Card> dealt_cards;
    for (int i = 0; i < num_cards; ++i) {
        dealt_cards.push_back(cards.back());
        cards.pop_back();
    }
    return dealt_cards;
}

} // namespace hokm
