#ifndef DECK_H
#define DECK_H

#include "card.h"
#include <vector>
#include <random>

namespace hokm {

class Deck {
public:
    std::vector<Card> cards;

    Deck();

    void shuffle(std::mt19937* rng = nullptr);
    std::vector<Card> deal(int num_cards);
};

} // namespace hokm

#endif // DECK_H
