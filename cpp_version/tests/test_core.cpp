#include <iostream>
#include <cassert>
#include "game_constants.h"
#include "card.h"
#include "deck.h"
#include "player.h"
#include "hokm.h"

using namespace hokm;

class DummyPlayer : public Player {
public:
    DummyPlayer(const std::string& name) : Player(name) {}

    std::vector<float> get_state() override {
        return std::vector<float>(STATE_DIM, 0.0f);
    }

    std::pair<Card, int> play_card(const std::string& lead_suit) override {
        std::vector<Card> legal;
        if (lead_suit.empty()) {
            legal = hand;
        } else {
            for (const auto& c : hand) {
                if (c.suit == lead_suit) {
                    legal.push_back(c);
                }
            }
            if (legal.empty()) legal = hand;
        }
        Card played = legal.front();
        return {played, card_to_index(played)};
    }

    double evaluate_play(const Card& card, const std::string& lead_suit, int round_num) override {
        return 0.0;
    }

    void store_experience(const std::vector<float>& state, int action, double reward, 
                          const std::vector<float>& next_state, bool done, 
                          bool rl_eligible, const std::vector<bool>& next_legal_mask) override {}

    void optimize_model() override {}

    double compute_terminal_reward() override {
        return 0.0;
    }

    void _sync_seats(Hokm* game) override {}
};

void test_card() {
    Card c("Hearts", "Ace");
    assert(c.suit == "Hearts");
    assert(c.rank == "Ace");
    assert(c.value == 14);
    
    Card c2("Spades", "10");
    assert(c2.to_string() == "10 of Spades");
    
    Card c3 = Card::from_string("King of Diamonds");
    assert(c3.suit == "Diamonds");
    assert(c3.rank == "King");
    assert(c3.value == 13);
    
    std::cout << "Card tests passed!" << std::endl;
}

void test_deck() {
    Deck d;
    assert(d.cards.size() == 52);
    
    auto cards = d.deal(5);
    assert(cards.size() == 5);
    assert(d.cards.size() == 47);
    
    std::cout << "Deck tests passed!" << std::endl;
}

void test_hokm() {
    std::vector<std::shared_ptr<Player>> players = {
        std::make_shared<DummyPlayer>("P1"),
        std::make_shared<DummyPlayer>("P2"),
        std::make_shared<DummyPlayer>("P3"),
        std::make_shared<DummyPlayer>("P4")
    };
    
    Hokm game(players);
    assert(game.players.size() == 4);
    assert(game.team1.size() == 2);
    assert(game.team2.size() == 2);
    
    auto hakem_cards = game.start_game();
    assert(hakem_cards.size() == 5);
    assert(game.hakem != nullptr);
    assert(game.hakem->hand.size() == 5);
    
    game.play_game(false);
    assert(game.scores[1] >= 7 || game.scores[2] >= 7);
    
    std::cout << "Hokm tests passed!" << std::endl;
}

int main() {
    test_card();
    test_deck();
    test_hokm();
    
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
