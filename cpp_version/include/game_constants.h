#ifndef GAME_CONSTANTS_H
#define GAME_CONSTANTS_H

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

namespace hokm {

const std::vector<std::string> SUITS = {"Hearts", "Diamonds", "Clubs", "Spades"};
const std::vector<std::string> RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"};

inline int get_rank_value(const std::string& rank) {
    for (size_t i = 0; i < RANKS.size(); ++i) {
        if (RANKS[i] == rank) {
            return static_cast<int>(i) + 2;
        }
    }
    throw std::invalid_argument("Invalid rank: " + rank);
}

const int STATE_DIM = 194;
const int ACTION_DIM = 52;

} // namespace hokm

#endif // GAME_CONSTANTS_H
