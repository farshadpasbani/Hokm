#include "hokm.h"
#include "enhanced_player.h"
#include <iostream>
#include <memory>
#include <chrono>

using namespace hokm;

int main(int argc, char* argv[]) {
    std::cout << "Starting Hokm C++ Training Loop..." << std::endl;
    
    int num_episodes = 1000;
    if (argc > 1) {
        num_episodes = std::stoi(argv[1]);
    }
    
    auto learner = std::make_shared<SharedNFSPLearner>();
    
    std::vector<std::shared_ptr<Player>> players = {
        std::make_shared<EnhancedPlayer>("Player 1", learner),
        std::make_shared<EnhancedPlayer>("Player 2", learner),
        std::make_shared<EnhancedPlayer>("Player 3", learner),
        std::make_shared<EnhancedPlayer>("Player 4", learner)
    };
    
    Hokm game(players, "", true); // minimal_logging = true
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int episode = 0; episode < num_episodes; ++episode) {
        game.play_game(false);
        
        if ((episode + 1) % 100 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            
            std::cout << "Episode " << (episode + 1) << "/" << num_episodes 
                      << " | Team 1 Score: " << game.scores[1] 
                      << " | Team 2 Score: " << game.scores[2] 
                      << " | Time: " << elapsed.count() << "s" << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
    
    return 0;
}
