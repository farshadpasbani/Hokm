#include "hokm.h"
#include <algorithm>
#include <iostream>
#include <chrono>

namespace hokm {

Hokm::Hokm(std::vector<std::shared_ptr<Player>> players, 
           const std::string& trick_csv_path, 
           bool minimal_logging, 
           std::mt19937* rng)
    : players(players), trick_csv_path(trick_csv_path), minimal_logging(minimal_logging), rng(rng),
      difficulty_level(1), game_count(0), round_count(0), trick_count(0), trick_starter_index(0),
      aborted_games(0) {
          
    if (players.size() != 4) {
        throw std::invalid_argument("Hokm requires exactly 4 players");
    }
    
    team1 = {players[0], players[2]};
    team2 = {players[1], players[3]};
    team_strategy = std::make_shared<TeamStrategy>();
    last_winning_team = team1;
    
    scores[1] = 0;
    scores[2] = 0;
    
    for (auto& p : players) {
        tricks_won[p] = 0;
        p->tricks_won_ptr = &tricks_won;
        p->team_strategy = team_strategy;
        
        if (p == team1[0] || p == team1[1]) {
            p->team = team1;
        } else {
            p->team = team2;
        }
    }
}

void Hokm::reset_players() {
    for (auto& p : players) {
        p->reset();
    }
    tricks_won.clear();
    for (auto& p : players) {
        tricks_won[p] = 0;
    }
    scores[1] = 0;
    scores[2] = 0;
    current_trick.clear();
    lead_suit = "";
    round_count = 0;
    trick_count = 0;
    trick_starter_index = 0;
    last_trick_winner = nullptr;
}

std::vector<Card> Hokm::start_game() {
    deck = Deck();
    deck.shuffle(rng);
    reset_players();
    
    if (!hakem) {
        if (rng) {
            std::uniform_int_distribution<int> dist(0, 3);
            hakem = players[dist(*rng)];
        } else {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::mt19937 default_rng(seed);
            std::uniform_int_distribution<int> dist(0, 3);
            hakem = players[dist(default_rng)];
        }
    }
    
    cards_played_this_hand.clear();
    void_map.clear();
    for (auto& p : players) {
        void_map[p] = std::set<std::string>();
        p->_sync_seats(this);
    }
    
    hakem_cards = deck.deal(5);
    hakem->hand = hakem_cards;
    sort_cards_in_hand(hakem->hand);
    hakem_cards = hakem->hand;

    return hakem_cards;
}

void Hokm::_sync_player_trick_context() {
    for (auto& p : players) {
        p->current_trick = current_trick;
        p->lead_suit = lead_suit;
    }
}

void Hokm::set_trump_suit(const std::string& trump_suit) {
    if (std::find(SUITS.begin(), SUITS.end(), trump_suit) == SUITS.end()) {
        throw std::invalid_argument("Invalid trump suit: " + trump_suit);
    }
    this->trump_suit = trump_suit;
    for (auto& p : players) {
        p->update_trump_suit(trump_suit);
    }
    
    for (auto& p : players) {
        int num_cards = (p == hakem) ? 8 : 13;
        p->draw(deck, num_cards);
    }
    for (auto& p : players) {
        sort_cards_in_hand(p->hand);
    }

    auto it = std::find(players.begin(), players.end(), hakem);
    trick_starter_index = std::distance(players.begin(), it);
    _sync_player_trick_context();
}

void Hokm::choose_trump_suit() {
    std::map<std::string, int> suit_counts;
    std::map<std::string, int> suit_values;
    
    for (const auto& suit : SUITS) {
        suit_counts[suit] = 0;
        suit_values[suit] = 0;
    }
    
    for (const auto& card : hakem->hand) {
        suit_counts[card.suit]++;
        suit_values[card.suit] += card.value * (card.value >= 10 ? 2 : 1);
    }
    
    std::string best_suit = SUITS[0];
    int max_score = -1;
    
    for (const auto& suit : SUITS) {
        int score = suit_counts[suit] * 10 + suit_values[suit];
        if (score > max_score) {
            max_score = score;
            best_suit = suit;
        }
    }
    
    set_trump_suit(best_suit);
}

std::shared_ptr<Player> Hokm::determine_trick_winner() {
    if (current_trick.empty()) return nullptr;
    
    Card winning_card = current_trick[0].second;
    std::shared_ptr<Player> winner = current_trick[0].first;
    
    bool has_trump = false;
    for (const auto& pair : current_trick) {
        if (pair.second.suit == trump_suit) {
            has_trump = true;
            break;
        }
    }
    
    for (size_t i = 1; i < current_trick.size(); ++i) {
        auto player = current_trick[i].first;
        auto card = current_trick[i].second;
        
        if (has_trump) {
            if (card.suit == trump_suit) {
                if (winning_card.suit != trump_suit || card.value > winning_card.value) {
                    winning_card = card;
                    winner = player;
                }
            }
        } else {
            if (card.suit == lead_suit) {
                if (winning_card.suit != lead_suit || card.value > winning_card.value) {
                    winning_card = card;
                    winner = player;
                }
            }
        }
    }
    
    tricks_won[winner]++;
    return winner;
}

std::shared_ptr<Player> Hokm::play_round() {
    current_trick.clear();
    lead_suit = "";
    
    auto hakem_it = std::find(players.begin(), players.end(), hakem);
    int hakem_index = std::distance(players.begin(), hakem_it);
    
    if (round_count == 0) {
        trick_starter_index = hakem_index;
    } else {
        if (!last_trick_winner) {
            throw std::runtime_error("last_trick_winner must be set before non-first tricks");
        }
        auto winner_it = std::find(players.begin(), players.end(), last_trick_winner);
        trick_starter_index = std::distance(players.begin(), winner_it);
    }
    
    int starting_player_index = trick_starter_index;
    std::shared_ptr<Player> current_player = players[starting_player_index];
    
    struct PendingTransition {
        std::shared_ptr<Player> player;
        std::vector<float> state;
        int action;
        double shaping_reward;
        bool rl_eligible;
    };
    std::vector<PendingTransition> pending;
    
    for (int i = 0; i < 4; ++i) {
        std::vector<float> state = current_player->get_state();
        
        auto result = current_player->play_card(lead_suit);
        Card card = result.first;
        int action_index = result.second;
        
        auto it = std::find(current_player->hand.begin(), current_player->hand.end(), card);
        if (it == current_player->hand.end()) {
            throw std::runtime_error("Player attempted to play a card not in hand");
        }
        
        current_player->hand.erase(it);
        current_trick.push_back({current_player, card});
        current_player->current_trick = current_trick;
        
        std::string prev_lead_suit = lead_suit;
        if (lead_suit.empty()) {
            lead_suit = card.suit;
        }
        
        current_player->actions_taken.push_back(action_index);
        current_player->played_cards_memory.insert(card.to_string());
        current_player->team_strategy->update_card_count(card);
        
        auto suit_it = std::find(SUITS.begin(), SUITS.end(), card.suit);
        if (suit_it != SUITS.end()) {
            current_player->played_suit_counts[std::distance(SUITS.begin(), suit_it)]++;
        }
        
        cards_played_this_hand.push_back(card);
        if (!prev_lead_suit.empty() && card.suit != prev_lead_suit) {
            void_map[current_player].insert(prev_lead_suit);
        }
        
        double shaping_reward = current_player->evaluate_play(card, lead_suit, round_count);
        
        pending.push_back({
            current_player,
            state,
            action_index,
            shaping_reward,
            current_player->last_rl_eligible
        });
        
        int current_player_index = (std::distance(players.begin(), std::find(players.begin(), players.end(), current_player)) + 1) % 4;
        current_player = players[current_player_index];
    }
    
    std::shared_ptr<Player> winner = determine_trick_winner();
    last_trick_winner = winner;
    trick_starter_index = std::distance(players.begin(), std::find(players.begin(), players.end(), winner));
    
    int team = (std::find(team1.begin(), team1.end(), winner) != team1.end()) ? 1 : 2;
    scores[team]++;
    
    bool game_over = scores[1] >= 7 || scores[2] >= 7;
    
    for (const auto& rec : pending) {
        auto p = rec.player;
        bool hand_empty = p->hand.empty();
        bool done = game_over || hand_empty;
        double reward = rec.shaping_reward;
        
        if (done) {
            reward += p->compute_terminal_reward();
        }
        
        std::vector<float> next_state = p->get_state();
        std::vector<bool> next_legal_mask(ACTION_DIM, false);
        
        if (!done) {
            for (const auto& c : p->hand) {
                next_legal_mask[card_to_index(c)] = true;
            }
        }
        
        p->store_experience(rec.state, rec.action, reward, next_state, done, rec.rl_eligible, next_legal_mask);
        p->optimize_model();
    }
    
    round_count++;
    return winner;
}

void Hokm::update_last_winning_team() {
    int team1_tricks = tricks_won[team1[0]] + tricks_won[team1[1]];
    int team2_tricks = tricks_won[team2[0]] + tricks_won[team2[1]];
    last_winning_team = (team1_tricks >= 7) ? team1 : team2;
}

void Hokm::rotate_hakem() {
    auto current_team = (std::find(team1.begin(), team1.end(), hakem) != team1.end()) ? team1 : team2;
    if (last_winning_team != current_team) {
        hakem = last_winning_team[0];
    } else {
        int current_idx = (current_team[0] == hakem) ? 0 : 1;
        int next_idx = (current_idx + 1) % 2;
        hakem = current_team[next_idx];
    }
}

void Hokm::play_game(bool save_excel_log) {
    game_count++;
    start_game();
    choose_trump_suit();
    round_count = 0;
    
    std::shared_ptr<Player> current_player = hakem;
    
    bool has_cards = true;
    while (has_cards) {
        has_cards = false;
        for (const auto& p : players) {
            if (!p->hand.empty()) {
                has_cards = true;
                break;
            }
        }
        
        if (!has_cards) break;
        
        try {
            std::shared_ptr<Player> winner = play_round();
            current_player = winner;
            if (scores[1] >= 7 || scores[2] >= 7) {
                break;
            }
        } catch (const std::exception& e) {
            aborted_games++;
            std::cerr << "Round error: " << e.what() << std::endl;
            break;
        }
    }
    
    update_last_winning_team();
    rotate_hakem();
}

std::shared_ptr<Player> Hokm::get_next_to_play() {
    if (current_trick.empty()) {
        return players[trick_starter_index];
    }
    auto last_player = current_trick.back().first;
    auto it = std::find(players.begin(), players.end(), last_player);
    int idx = std::distance(players.begin(), it);
    return players[(idx + 1) % 4];
}

std::vector<Card> Hokm::legal_cards_for_player(std::shared_ptr<Player> player) {
    if (lead_suit.empty()) {
        return player->hand;
    }
    std::vector<Card> following;
    for (const auto& c : player->hand) {
        if (c.suit == lead_suit) {
            following.push_back(c);
        }
    }
    return following.empty() ? player->hand : following;
}

std::string Hokm::apply_play(std::shared_ptr<Player> player, const Card& card) {
    if (player != get_next_to_play()) {
        return "Not this player's turn";
    }
    auto it = std::find(player->hand.begin(), player->hand.end(), card);
    if (it == player->hand.end()) {
        return "Card not in hand";
    }
    
    auto legal = legal_cards_for_player(player);
    if (std::find(legal.begin(), legal.end(), card) == legal.end()) {
        return "Illegal card for this trick";
    }
    
    player->hand.erase(it);
    current_trick.push_back({player, card});
    
    std::string prev_lead_suit = lead_suit;
    if (lead_suit.empty()) {
        lead_suit = card.suit;
    }
    
    cards_played_this_hand.push_back(card);
    if (!prev_lead_suit.empty() && card.suit != prev_lead_suit) {
        void_map[player].insert(prev_lead_suit);
    }
    
    _sync_player_trick_context();
    return "";
}

std::shared_ptr<Player> Hokm::resolve_trick_if_complete() {
    if (current_trick.size() < 4) {
        return nullptr;
    }
    std::shared_ptr<Player> winner = determine_trick_winner();
    last_trick_winner = winner;
    trick_starter_index = std::distance(
        players.begin(), std::find(players.begin(), players.end(), winner)
    );
    int team = (std::find(team1.begin(), team1.end(), winner) != team1.end()) ? 1 : 2;
    scores[team]++;
    current_trick.clear();
    lead_suit.clear();
    round_count++;
    _sync_player_trick_context();
    return winner;
}

bool Hokm::is_hand_over() const {
    if (scores.at(1) >= 7 || scores.at(2) >= 7) {
        return true;
    }
    for (const auto& p : players) {
        if (!p->hand.empty()) {
            return false;
        }
    }
    return true;
}

} // namespace hokm
