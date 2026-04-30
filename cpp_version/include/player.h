#ifndef PLAYER_H
#define PLAYER_H

#include "card.h"
#include <string>
#include <vector>
#include <memory>
#include <set>
#include <map>

namespace hokm {

class Deck;
class Hokm;
class TeamStrategy; // Forward declaration

class Player {
public:
    std::string name;
    std::vector<Card> hand;
    std::vector<std::shared_ptr<Player>> team;
    std::map<std::shared_ptr<Player>, int>* tricks_won_ptr;
    std::shared_ptr<TeamStrategy> team_strategy;
    
    std::vector<std::pair<std::shared_ptr<Player>, Card>> current_trick;
    std::string lead_suit;
    std::string trump_suit;
    
    std::vector<int> actions_taken;
    std::set<std::string> played_cards_memory;
    std::vector<int> played_suit_counts;
    
    double total_reward;
    bool last_rl_eligible;

    Player(const std::string& name);
    virtual ~Player() = default;

    virtual void reset();
    virtual void draw(Deck& deck, int num_cards);
    virtual void update_trump_suit(const std::string& trump_suit);
    
    // Returns {state_vector}
    virtual std::vector<float> get_state() = 0;
    
    // Returns {card, action_index}
    virtual std::pair<Card, int> play_card(const std::string& lead_suit) = 0;
    
    // Returns shaping reward
    virtual double evaluate_play(const Card& card, const std::string& lead_suit, int round_num) = 0;
    
    // RL training methods
    virtual void store_experience(const std::vector<float>& state, int action, double reward, 
                                  const std::vector<float>& next_state, bool done, 
                                  bool rl_eligible, const std::vector<bool>& next_legal_mask) = 0;
    virtual void optimize_model() = 0;
    virtual double compute_terminal_reward() = 0;
    
    virtual void _sync_seats(Hokm* game) = 0;
};

class TeamStrategy {
public:
    std::map<std::string, int> card_count;
    
    TeamStrategy();
    void update_card_count(const Card& card);
    bool should_conserve_trump(std::shared_ptr<Player> player);
};

} // namespace hokm

#endif // PLAYER_H
