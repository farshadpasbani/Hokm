#ifndef ENHANCED_PLAYER_H
#define ENHANCED_PLAYER_H

#include "player.h"
#include "game_constants.h"
#include <torch/torch.h>
#include <deque>
#include <random>

namespace hokm {

// Forward declarations
class QNetwork;
class AveragePolicyNetwork;

struct Experience {
    std::vector<float> state;
    int action;
    double reward;
    std::vector<float> next_state;
    bool done;
    bool rl_eligible;
    std::vector<bool> next_legal_mask;
};

class UniformReplayMemory {
public:
    int capacity;
    std::vector<Experience> memory;
    int position;
    std::mt19937 rng;

    UniformReplayMemory(int capacity);
    void push(const Experience& exp);
    std::vector<Experience> sample(int batch_size);
    size_t size() const;
};

class QNetworkImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr}, ln3{nullptr};

    QNetworkImpl(int input_dim, int output_dim);
    torch::Tensor _embed(torch::Tensor x);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor q_values_at_indices(torch::Tensor x, const std::vector<int>& indices);
};
TORCH_MODULE(QNetwork);

class AveragePolicyNetworkImpl : public torch::nn::Module {
public:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};

    AveragePolicyNetworkImpl(int input_dim, int output_dim);
    torch::Tensor _embed(torch::Tensor x);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor logits_at_indices(torch::Tensor x, const std::vector<int>& indices);
};
TORCH_MODULE(AveragePolicyNetwork);

class SharedNFSPLearner {
public:
    QNetwork q_network;
    QNetwork target_q_network;
    AveragePolicyNetwork policy_network;
    
    std::shared_ptr<torch::optim::Adam> q_optimizer;
    std::shared_ptr<torch::optim::Adam> policy_optimizer;
    
    UniformReplayMemory rl_memory;
    UniformReplayMemory sl_memory;
    
    double gamma;
    int batch_size;
    int target_update_freq;
    int steps_done;
    int env_steps;
    int optimize_every_steps;
    double latest_q_loss;
    double latest_policy_loss;
    
    torch::Device device;

    SharedNFSPLearner(int state_dim = STATE_DIM, int action_dim = ACTION_DIM, 
                      int rl_capacity = 100000, int sl_capacity = 1000000);
                      
    void push_transition(const Experience& exp, bool is_best_response);
    void optimize_q();
    void optimize_policy();
    bool save_models(const std::string& directory) const;
    bool load_models(const std::string& directory);
};

class EnhancedPlayer : public Player {
public:
    std::shared_ptr<SharedNFSPLearner> learner;
    double epsilon;
    double eta;
    std::string reward_mode;
    std::mt19937 rng;
    
    bool is_best_response;
    
    std::shared_ptr<Player> LHO;
    std::shared_ptr<Player> RHO;
    std::shared_ptr<Player> partner;
    std::string last_strategy_reason;
    bool learning_enabled;

    EnhancedPlayer(const std::string& name, std::shared_ptr<SharedNFSPLearner> learner, 
                   double epsilon = 0.1, double eta = 0.1, const std::string& reward_mode = "mixed");

    void reset() override;
    std::vector<float> get_state() override;
    std::pair<Card, int> play_card(const std::string& lead_suit) override;
    double evaluate_play(const Card& card, const std::string& lead_suit, int round_num) override;
    void store_experience(const std::vector<float>& state, int action, double reward, 
                          const std::vector<float>& next_state, bool done, 
                          bool rl_eligible, const std::vector<bool>& next_legal_mask) override;
    void optimize_model() override;
    double compute_terminal_reward() override;
    void _sync_seats(Hokm* game) override;
};

} // namespace hokm

#endif // ENHANCED_PLAYER_H
