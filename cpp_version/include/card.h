#ifndef CARD_H
#define CARD_H

#include <string>
#include <functional>

namespace hokm {

class Card {
public:
    std::string suit;
    std::string rank;
    int value;

    Card();
    Card(const std::string& suit, const std::string& rank);

    std::string to_string() const;
    
    bool operator==(const Card& other) const;
    bool operator!=(const Card& other) const;

    static Card from_string(const std::string& card_string);
};

int card_to_index(const Card& card);
Card index_to_card(int index);

} // namespace hokm

namespace std {
    template <>
    struct hash<hokm::Card> {
        size_t operator()(const hokm::Card& card) const {
            return hash<string>()(card.suit) ^ (hash<string>()(card.rank) << 1);
        }
    };
}

#endif // CARD_H
