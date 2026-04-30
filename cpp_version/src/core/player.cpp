#include "player.h"
#include "deck.h"
#include "game_constants.h"

namespace hokm {

Player::Player(const std::string& name) : name(name), tricks_won_ptr(nullptr), total_reward(0.0), last_rl_eligible(true) {
    played_suit_counts.resize(4, 0);
}

void Player::reset() {
    hand.clear();
    current_trick.clear();
    lead_suit = "";
    trump_suit = "";
    actions_taken.clear();
    played_cards_memory.clear();
    std::fill(played_suit_counts.begin(), played_suit_counts.end(), 0);
    total_reward = 0.0;
    last_rl_eligible = true;
}

void Player::draw(Deck& deck, int num_cards) {
    std::vector<Card> drawn = deck.deal(num_cards);
    hand.insert(hand.end(), drawn.begin(), drawn.end());
}

void Player::update_trump_suit(const std::string& trump_suit) {
    this->trump_suit = trump_suit;
}

TeamStrategy::TeamStrategy() {
    for (const auto& suit : SUITS) {
        card_count[suit] = 13;
    }
}

void TeamStrategy::update_card_count(const Card& card) {
    if (card_count[card.suit] > 0) {
        card_count[card.suit]--;
    }
}

bool TeamStrategy::should_conserve_trump(std::shared_ptr<Player> player) {
    if (!player || player->trump_suit.empty()) return false;
    
    int remaining_trump = 0;
    for (const auto& card : player->hand) {
        if (card.suit == player->trump_suit) {
            remaining_trump++;
        }
    }
    
    int played_trump = 13 - card_count[player->trump_suit];
    return remaining_trump < 3 && played_trump < 6;
}

} // namespace hokm
