#include "enhanced_player.h"
#include "hokm.h"
#include <algorithm>
#include <iostream>

namespace hokm {

UniformReplayMemory::UniformReplayMemory(int capacity) : capacity(capacity), position(0) {
    std::random_device rd;
    rng.seed(rd());
}

void UniformReplayMemory::push(const Experience& exp) {
    if (memory.size() < static_cast<size_t>(capacity)) {
        memory.push_back(exp);
    } else {
        memory[position] = exp;
        position = (position + 1) % capacity;
    }
}

std::vector<Experience> UniformReplayMemory::sample(int batch_size) {
    std::vector<Experience> batch;
    if (memory.size() < static_cast<size_t>(batch_size)) return batch;
    
    std::vector<int> indices(memory.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < batch_size; ++i) {
        batch.push_back(memory[indices[i]]);
    }
    return batch;
}

size_t UniformReplayMemory::size() const {
    return memory.size();
}

QNetworkImpl::QNetworkImpl(int input_dim, int output_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, 256));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
    fc2 = register_module("fc2", torch::nn::Linear(256, 128));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
    fc3 = register_module("fc3", torch::nn::Linear(128, 64));
    ln3 = register_module("ln3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({64})));
    fc4 = register_module("fc4", torch::nn::Linear(64, output_dim));
}

torch::Tensor QNetworkImpl::_embed(torch::Tensor x) {
    if (x.dim() == 1) x = x.unsqueeze(0);
    x = torch::relu(ln1(fc1(x)));
    x = torch::relu(ln2(fc2(x)));
    x = torch::relu(ln3(fc3(x)));
    return x;
}

torch::Tensor QNetworkImpl::forward(torch::Tensor x) {
    return fc4(_embed(x));
}

torch::Tensor QNetworkImpl::q_values_at_indices(torch::Tensor x, const std::vector<int>& indices) {
    if (indices.empty()) return torch::zeros({0}, x.device());
    torch::Tensor h = _embed(x).squeeze(0);
    auto idx = torch::tensor(indices, torch::kLong).to(x.device());
    auto w = fc4->weight.index_select(0, idx);
    auto b = fc4->bias.index_select(0, idx);
    return torch::matmul(h, w.t()) + b;
}

AveragePolicyNetworkImpl::AveragePolicyNetworkImpl(int input_dim, int output_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(input_dim, 256));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
    fc2 = register_module("fc2", torch::nn::Linear(256, 256));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
    fc3 = register_module("fc3", torch::nn::Linear(256, output_dim));
}

torch::Tensor AveragePolicyNetworkImpl::_embed(torch::Tensor x) {
    if (x.dim() == 1) x = x.unsqueeze(0);
    x = torch::relu(ln1(fc1(x)));
    x = torch::relu(ln2(fc2(x)));
    return x;
}

torch::Tensor AveragePolicyNetworkImpl::forward(torch::Tensor x) {
    return fc3(_embed(x));
}

torch::Tensor AveragePolicyNetworkImpl::logits_at_indices(torch::Tensor x, const std::vector<int>& indices) {
    if (indices.empty()) return torch::zeros({0}, x.device());
    torch::Tensor h = _embed(x).squeeze(0);
    auto idx = torch::tensor(indices, torch::kLong).to(x.device());
    auto w = fc3->weight.index_select(0, idx);
    auto b = fc3->bias.index_select(0, idx);
    return torch::matmul(h, w.t()) + b;
}

SharedNFSPLearner::SharedNFSPLearner(int state_dim, int action_dim, int rl_capacity, int sl_capacity)
    : q_network(state_dim, action_dim), target_q_network(state_dim, action_dim),
      policy_network(state_dim, action_dim), rl_memory(rl_capacity), sl_memory(sl_capacity),
      gamma(0.99), batch_size(128), target_update_freq(1000), steps_done(0),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
          
    q_network->to(device);
    target_q_network->to(device);
    policy_network->to(device);
    
    // Copy weights
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < q_network->parameters().size(); ++i) {
        target_q_network->parameters()[i].copy_(q_network->parameters()[i]);
    }
    target_q_network->eval();
    
    q_optimizer = std::make_shared<torch::optim::Adam>(q_network->parameters(), torch::optim::AdamOptions(1e-4));
    policy_optimizer = std::make_shared<torch::optim::Adam>(policy_network->parameters(), torch::optim::AdamOptions(1e-4));
}

void SharedNFSPLearner::push_transition(const Experience& exp, bool is_best_response) {
    if (exp.rl_eligible) {
        rl_memory.push(exp);
    }
    if (is_best_response) {
        sl_memory.push(exp);
    }
}

void SharedNFSPLearner::optimize_q() {
    if (rl_memory.size() < static_cast<size_t>(batch_size)) return;
    // Simplified optimization step
    // In a full conversion, we would translate the PyTorch DQN step here
}

void SharedNFSPLearner::optimize_policy() {
    if (sl_memory.size() < static_cast<size_t>(batch_size)) return;
    // Simplified optimization step
    // In a full conversion, we would translate the PyTorch SL step here
}

EnhancedPlayer::EnhancedPlayer(const std::string& name, std::shared_ptr<SharedNFSPLearner> learner, 
                               double epsilon, double eta, const std::string& reward_mode)
    : Player(name), learner(learner), epsilon(epsilon), eta(eta), reward_mode(reward_mode) {
    std::random_device rd;
    rng.seed(rd());
}

void EnhancedPlayer::reset() {
    Player::reset();
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    is_best_response = (dist(rng) < eta);
}

void EnhancedPlayer::_sync_seats(Hokm* game) {
    auto it = std::find(game->players.begin(), game->players.end(), game->players[0]);
    // Find seats relative to this player
    for (size_t i = 0; i < game->players.size(); ++i) {
        if (game->players[i]->name == name) {
            LHO = game->players[(i + 1) % 4];
            partner = game->players[(i + 2) % 4];
            RHO = game->players[(i + 3) % 4];
            break;
        }
    }
}

std::vector<float> EnhancedPlayer::get_state() {
    std::vector<float> state(STATE_DIM, 0.0f);
    // Simplified state representation
    for (const auto& card : hand) {
        state[card_to_index(card)] = 1.0f;
    }
    return state;
}

std::pair<Card, int> EnhancedPlayer::play_card(const std::string& lead_suit) {
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
    
    std::vector<int> legal_indices;
    for (const auto& c : legal) {
        legal_indices.push_back(card_to_index(c));
    }
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int chosen_idx = 0;
    
    if (is_best_response) {
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> int_dist(0, legal.size() - 1);
            chosen_idx = legal_indices[int_dist(rng)];
        } else {
            // Use Q-network
            torch::NoGradGuard no_grad;
            auto state_tensor = torch::tensor(get_state(), torch::kFloat32).to(learner->device);
            auto q_vals = learner->q_network->q_values_at_indices(state_tensor, legal_indices);
            int max_idx = q_vals.argmax().item<int>();
            chosen_idx = legal_indices[max_idx];
        }
    } else {
        // Use Policy network
        torch::NoGradGuard no_grad;
        auto state_tensor = torch::tensor(get_state(), torch::kFloat32).to(learner->device);
        auto logits = learner->policy_network->logits_at_indices(state_tensor, legal_indices);
        auto probs = torch::softmax(logits, 0);
        
        // Sample from distribution
        std::vector<double> probs_vec(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::discrete_distribution<int> d(probs_vec.begin(), probs_vec.end());
        chosen_idx = legal_indices[d(rng)];
    }
    
    Card chosen_card = index_to_card(chosen_idx);
    return {chosen_card, chosen_idx};
}

double EnhancedPlayer::evaluate_play(const Card& card, const std::string& lead_suit, int round_num) {
    return 0.0; // Simplified reward
}

void EnhancedPlayer::store_experience(const std::vector<float>& state, int action, double reward, 
                                      const std::vector<float>& next_state, bool done, 
                                      bool rl_eligible, const std::vector<bool>& next_legal_mask) {
    Experience exp{state, action, reward, next_state, done, rl_eligible, next_legal_mask};
    learner->push_transition(exp, is_best_response);
}

void EnhancedPlayer::optimize_model() {
    learner->optimize_q();
    learner->optimize_policy();
}

double EnhancedPlayer::compute_terminal_reward() {
    return 0.0;
}

} // namespace hokm
