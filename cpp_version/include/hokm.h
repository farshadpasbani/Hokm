#ifndef HOKM_H
#define HOKM_H

#include "player.h"
#include "deck.h"
#include "game_constants.h"
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <random>
#include <string>

namespace hokm {

class Hokm {
public:
    std::vector<std::shared_ptr<Player>> players;
    std::string trick_csv_path;
    bool minimal_logging;
    std::mt19937* rng;
    
    Deck deck;
    std::vector<std::pair<std::shared_ptr<Player>, Card>> current_trick;
    std::string lead_suit;
    std::string trump_suit;
    std::shared_ptr<Player> hakem;
    
    std::map<int, int> scores;
    int difficulty_level;
    std::vector<Card> hakem_cards;
    
    int game_count;
    int round_count;
    int trick_count;
    int trick_starter_index;
    std::shared_ptr<Player> last_trick_winner;
    
    std::vector<std::shared_ptr<Player>> team1;
    std::vector<std::shared_ptr<Player>> team2;
    std::shared_ptr<TeamStrategy> team_strategy;
    std::map<std::shared_ptr<Player>, int> tricks_won;
    std::vector<std::shared_ptr<Player>> last_winning_team;
    
    std::string session_id;
    
    std::vector<Card> cards_played_this_hand;
    std::map<std::shared_ptr<Player>, std::set<std::string>> void_map;
    
    int aborted_games;

    Hokm(std::vector<std::shared_ptr<Player>> players, 
         const std::string& trick_csv_path = "", 
         bool minimal_logging = false, 
         std::mt19937* rng = nullptr);
         
    void reset_players();
    std::vector<Card> start_game();
    void set_trump_suit(const std::string& trump_suit);
    void choose_trump_suit();
    std::shared_ptr<Player> play_round();
    void _sync_player_trick_context();
    std::shared_ptr<Player> get_next_to_play();
    std::vector<Card> legal_cards_for_player(std::shared_ptr<Player> player);
    std::string apply_play(std::shared_ptr<Player> player, const Card& card);
    std::shared_ptr<Player> determine_trick_winner();
    void update_last_winning_team();
    void rotate_hakem();
    void play_game(bool save_excel_log = true);
};

} // namespace hokm

#endif // HOKM_H
