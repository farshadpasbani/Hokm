#include "enhanced_player.h"
#include "hokm.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>

namespace {

using hokm::Card;
using hokm::Player;
using std::shared_ptr;

std::vector<Card> legal_cards_from_hand(
    const std::vector<Card>& hand,
    const std::string& lead_suit
) {
    if (lead_suit.empty()) return hand;
    std::vector<Card> follow;
    for (const auto& c : hand) {
        if (c.suit == lead_suit) follow.push_back(c);
    }
    return follow.empty() ? hand : follow;
}

std::pair<shared_ptr<Player>, Card> current_winner(
    const std::vector<std::pair<shared_ptr<Player>, Card>>& trick,
    const std::string& lead_suit,
    const std::string& trump_suit
) {
    auto winner = trick.front();
    bool has_trump = false;
    for (const auto& pc : trick) {
        if (pc.second.suit == trump_suit) {
            has_trump = true;
            break;
        }
    }
    for (size_t i = 1; i < trick.size(); ++i) {
        const auto& card = trick[i].second;
        if (has_trump) {
            if (card.suit == trump_suit &&
                (winner.second.suit != trump_suit || card.value > winner.second.value)) {
                winner = trick[i];
            }
        } else {
            if (card.suit == lead_suit &&
                (winner.second.suit != lead_suit || card.value > winner.second.value)) {
                winner = trick[i];
            }
        }
    }
    return winner;
}

double card_value_norm(const Card& c) {
    return std::max(0.0, std::min(1.0, (static_cast<double>(c.value) - 2.0) / 12.0));
}

double estimate_individual_beat_probability(
    const Card& candidate,
    const std::string& lead_suit,
    const std::string& trump_suit,
    int remaining_trump,
    int remaining_cards
) {
    if (remaining_cards <= 0) return 0.0;
    const bool cand_is_trump = (!trump_suit.empty() && candidate.suit == trump_suit);
    const bool cand_is_lead = (!lead_suit.empty() && candidate.suit == lead_suit);
    const double highers = std::max(0.0, (14.0 - static_cast<double>(candidate.value)) / 13.0);
    const double trump_density = std::max(
        0.0, std::min(1.0, static_cast<double>(remaining_trump) / std::max(1, remaining_cards))
    );

    if (cand_is_trump) {
        // Trump can only be beaten by higher trump.
        return std::min(0.95, highers * (0.45 + 0.55 * trump_density));
    }
    if (cand_is_lead) {
        // Lead-suit winners can be beaten by higher lead OR trump cuts.
        const double higher_lead = highers * 0.55;
        const double trump_cut = trump_density * 0.45;
        return std::min(0.95, higher_lead + trump_cut);
    }
    // Off-suit non-trump almost never wins unless everyone is voiding weirdly.
    return 0.98;
}

double estimate_trick_win_probability(
    const Card& candidate,
    const std::vector<std::pair<shared_ptr<Player>, Card>>& trick,
    const std::string& lead_suit,
    const std::string& trump_suit,
    const shared_ptr<Player>& partner
) {
    const int cards_already_in_trick = static_cast<int>(trick.size());
    int remaining_to_play = std::max(0, 3 - cards_already_in_trick);

    if (cards_already_in_trick == 0) {
        // If we lead, assume all 3 others can still respond.
        remaining_to_play = 3;
    }

    std::vector<std::pair<shared_ptr<Player>, Card>> with_candidate = trick;
    with_candidate.push_back({nullptr, candidate});
    const std::string effective_lead = lead_suit.empty() ? candidate.suit : lead_suit;
    auto current = current_winner(with_candidate, effective_lead, trump_suit);
    const bool candidate_currently_winning = (current.second == candidate);

    // If partner already winning and we don't overtake, team chance is high and
    // should push us toward conserving power cards.
    const bool partner_currently_winning =
        (!trick.empty() && partner && current.first && current.first->name == partner->name);
    if (partner_currently_winning && !candidate_currently_winning) {
        return 0.82;
    }

    if (!candidate_currently_winning) {
        return 0.04;
    }

    int remaining_trump = 13;
    int remaining_cards = 52 - static_cast<int>(trick.size()) - 1;
    if (remaining_cards <= 0) remaining_cards = 1;

    const double beat_p = estimate_individual_beat_probability(
        candidate, effective_lead, trump_suit, remaining_trump, remaining_cards
    );
    const double survive = std::pow(std::max(0.0, 1.0 - beat_p), remaining_to_play);
    return std::max(0.01, std::min(0.99, survive));
}

double basic_strategy_score(
    const Card& card,
    const std::vector<Card>& legal,
    const std::vector<Card>& full_hand,
    const std::vector<std::pair<shared_ptr<Player>, Card>>& trick,
    const std::string& lead_suit,
    const std::string& trump_suit,
    const shared_ptr<Player>& partner
) {
    const bool leading = trick.empty();
    const bool is_trump = (!trump_suit.empty() && card.suit == trump_suit);
    int trump_count = 0;
    int suit_count = 0;
    for (const auto& c : full_hand) {
        if (c.suit == trump_suit) trump_count++;
        if (c.suit == card.suit) suit_count++;
    }

    const double win_prob = estimate_trick_win_probability(
        card, trick, lead_suit, trump_suit, partner
    );
    const double strength = card_value_norm(card);
    double score = 0.0;

    if (leading) {
        // Probability-first lead logic:
        // - favor cards with higher expected control of trick
        // - avoid spending high trump when win probability is weak
        score += win_prob * 12.0;
        score += suit_count * 0.7;
        if (!is_trump) score += 1.8;
        if (is_trump && trump_count <= 3) score -= 3.0;
        // Wasting high cards on low-probability spots is strongly punished.
        score -= strength * (1.0 - win_prob) * 8.5;
        return score;
    }

    auto cw = current_winner(trick, lead_suit, trump_suit);
    const bool partner_winning = (cw.first && partner && cw.first->name == partner->name);
    const bool must_follow = !lead_suit.empty() && card.suit == lead_suit;

    if (partner_winning) {
        // Partner control: preserve equity by dumping low.
        score += win_prob * 2.0;
        score -= strength * 7.5;
        if (is_trump) score -= 3.8;
        return score;
    }

    bool card_can_win = false;
    if (must_follow) {
        if (cw.second.suit == lead_suit && card.value > cw.second.value) card_can_win = true;
    } else if (!lead_suit.empty() && card.suit == trump_suit) {
        if (cw.second.suit != trump_suit || card.value > cw.second.value) card_can_win = true;
    }

    // Expected value with explicit anti-waste behavior.
    if (card_can_win) score += 2.5;
    score += win_prob * 14.0;
    score -= strength * (1.0 - win_prob) * 10.0;   // key anti-waste term
    if (is_trump && win_prob < 0.55) score -= 2.8; // don't burn trump on thin odds
    if (!card_can_win) score -= 2.5;
    return score;
}

int select_by_strategy(
    const std::vector<Card>& legal,
    const std::vector<Card>& full_hand,
    const std::vector<std::pair<shared_ptr<Player>, Card>>& trick,
    const std::string& lead_suit,
    const std::string& trump_suit,
    const shared_ptr<Player>& partner
) {
    int best_i = 0;
    double best_score = -1e18;
    for (size_t i = 0; i < legal.size(); ++i) {
        double s = basic_strategy_score(
            legal[i], legal, full_hand, trick, lead_suit, trump_suit, partner
        );
        if (s > best_score) {
            best_score = s;
            best_i = static_cast<int>(i);
        }
    }
    return best_i;
}

std::string explain_basic_strategy(
    const Card& card,
    const std::vector<Card>& full_hand,
    const std::vector<std::pair<shared_ptr<Player>, Card>>& trick,
    const std::string& lead_suit,
    const std::string& trump_suit,
    const shared_ptr<Player>& partner
) {
    const double win_prob = estimate_trick_win_probability(
        card, trick, lead_suit, trump_suit, partner
    );
    const bool leading = trick.empty();
    if (leading) {
        if ((!trump_suit.empty() && card.suit == trump_suit) && win_prob < 0.55) {
            return "Lead selected with low confidence; preserves stronger non-trump alternatives";
        }
        return "Lead chosen by highest expected trick control probability";
    }
    auto cw = current_winner(trick, lead_suit, trump_suit);
    const bool partner_winning = (cw.first && partner && cw.first->name == partner->name);
    if (partner_winning) {
        return "Partner winning trick: conserve high cards and avoid waste";
    }
    if (card.suit == lead_suit && cw.second.suit == lead_suit && card.value > cw.second.value) {
        return "Following lead: selected near-minimum winner with favorable hold probability";
    }
    if (!trump_suit.empty() && card.suit == trump_suit &&
        (cw.second.suit != trump_suit || card.value > cw.second.value)) {
        return "Trump used because win probability outweighs conservation risk";
    }
    return "Low win probability state: discard low-cost card, preserve high cards";
}

} // namespace

namespace hokm {

namespace {
std::mutex g_hokm_torch_mutex;
}

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
      gamma(0.99), batch_size(128), target_update_freq(1000), steps_done(0), env_steps(0), optimize_every_steps(32),
      latest_q_loss(0.0), latest_policy_loss(0.0),
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
    env_steps++;
    if (exp.rl_eligible) {
        rl_memory.push(exp);
    }
    if (is_best_response) {
        sl_memory.push(exp);
    }
}

void SharedNFSPLearner::optimize_q() {
    if (env_steps % optimize_every_steps != 0) return;
    if (rl_memory.size() < static_cast<size_t>(batch_size)) return;
    auto batch = rl_memory.sample(batch_size);
    if (batch.empty()) return;

    std::vector<torch::Tensor> states_v;
    std::vector<torch::Tensor> next_states_v;
    std::vector<int64_t> actions_v;
    std::vector<float> rewards_v;
    std::vector<float> done_v;
    std::vector<torch::Tensor> next_masks_v;
    states_v.reserve(batch.size());
    next_states_v.reserve(batch.size());
    next_masks_v.reserve(batch.size());

    for (const auto& e : batch) {
        states_v.push_back(torch::tensor(e.state, torch::kFloat32));
        next_states_v.push_back(torch::tensor(e.next_state, torch::kFloat32));
        actions_v.push_back(static_cast<int64_t>(e.action));
        rewards_v.push_back(static_cast<float>(e.reward));
        done_v.push_back(e.done ? 1.0f : 0.0f);
        std::vector<float> maskf(e.next_legal_mask.size(), 0.0f);
        for (size_t i = 0; i < e.next_legal_mask.size(); ++i) {
            maskf[i] = e.next_legal_mask[i] ? 1.0f : 0.0f;
        }
        next_masks_v.push_back(torch::tensor(maskf, torch::kFloat32));
    }

    std::lock_guard<std::mutex> lk(g_hokm_torch_mutex);
    try {
        auto states = torch::stack(states_v).to(device);
        auto next_states = torch::stack(next_states_v).to(device);
        auto actions = torch::tensor(actions_v, torch::kLong).to(device);
        auto rewards = torch::tensor(rewards_v, torch::kFloat32).to(device);
        auto dones = torch::tensor(done_v, torch::kFloat32).to(device);
        auto next_masks = torch::stack(next_masks_v).to(device);

        auto q_all = q_network->forward(states);
        auto q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1);

        auto next_q_all = target_q_network->forward(next_states);
        auto neg_inf = torch::full_like(next_q_all, -1e9);
        auto masked_next_q = torch::where(next_masks > 0.5, next_q_all, neg_inf);
        auto next_q = std::get<0>(masked_next_q.max(1));
        next_q = torch::where(torch::isinf(next_q), torch::zeros_like(next_q), next_q);

        auto target = rewards + (1.0 - dones) * static_cast<float>(gamma) * next_q;
        auto loss = torch::smooth_l1_loss(q_taken, target.detach());
        latest_q_loss = loss.item<double>();

        q_optimizer->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(q_network->parameters(), 5.0);
        q_optimizer->step();

        steps_done++;
        if (steps_done % target_update_freq == 0) {
            torch::NoGradGuard no_grad;
            auto src = q_network->named_parameters();
            auto dst = target_q_network->named_parameters(true);
            for (const auto& p : src) {
                dst[p.key()].copy_(p.value());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Hokm] optimize_q failed: " << e.what() << std::endl;
        latest_q_loss = std::numeric_limits<double>::quiet_NaN();
    } catch (...) {
        std::cerr << "[Hokm] optimize_q failed: unknown exception" << std::endl;
        latest_q_loss = std::numeric_limits<double>::quiet_NaN();
    }
}

void SharedNFSPLearner::optimize_policy() {
    if (env_steps % optimize_every_steps != 0) return;
    if (sl_memory.size() < static_cast<size_t>(batch_size)) return;
    auto batch = sl_memory.sample(batch_size);
    if (batch.empty()) return;

    std::vector<torch::Tensor> states_v;
    std::vector<int64_t> actions_v;
    states_v.reserve(batch.size());
    actions_v.reserve(batch.size());
    for (const auto& e : batch) {
        states_v.push_back(torch::tensor(e.state, torch::kFloat32));
        actions_v.push_back(static_cast<int64_t>(e.action));
    }
    std::lock_guard<std::mutex> lk(g_hokm_torch_mutex);
    try {
        auto states = torch::stack(states_v).to(device);
        auto actions = torch::tensor(actions_v, torch::kLong).to(device);

        auto logits = policy_network->forward(states);
        auto loss = torch::nn::functional::cross_entropy(logits, actions);
        latest_policy_loss = loss.item<double>();
        policy_optimizer->zero_grad();
        loss.backward();
        torch::nn::utils::clip_grad_norm_(policy_network->parameters(), 5.0);
        policy_optimizer->step();
    } catch (const std::exception& e) {
        std::cerr << "[Hokm] optimize_policy failed: " << e.what() << std::endl;
        latest_policy_loss = std::numeric_limits<double>::quiet_NaN();
    } catch (...) {
        std::cerr << "[Hokm] optimize_policy failed: unknown exception" << std::endl;
        latest_policy_loss = std::numeric_limits<double>::quiet_NaN();
    }
}

bool SharedNFSPLearner::save_models(const std::string& directory) const {
    try {
        std::lock_guard<std::mutex> lk(g_hokm_torch_mutex);
        std::filesystem::create_directories(directory);
        torch::save(q_network, directory + "/q_network.pt");
        torch::save(target_q_network, directory + "/target_q_network.pt");
        torch::save(policy_network, directory + "/policy_network.pt");
        return true;
    } catch (...) {
        return false;
    }
}

bool SharedNFSPLearner::load_models(const std::string& directory) {
    try {
        std::lock_guard<std::mutex> lk(g_hokm_torch_mutex);
        torch::load(q_network, directory + "/q_network.pt", device);
        torch::load(target_q_network, directory + "/target_q_network.pt", device);
        torch::load(policy_network, directory + "/policy_network.pt", device);
        q_network->to(device);
        target_q_network->to(device);
        policy_network->to(device);
        return true;
    } catch (...) {
        return false;
    }
}

EnhancedPlayer::EnhancedPlayer(const std::string& name, std::shared_ptr<SharedNFSPLearner> learner, 
                               double epsilon, double eta, const std::string& reward_mode)
    : Player(name), learner(learner), epsilon(epsilon), eta(eta), reward_mode(reward_mode), last_strategy_reason("init"), learning_enabled(true) {
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
    std::vector<Card> legal = legal_cards_from_hand(hand, lead_suit);
    
    std::vector<int> legal_indices;
    for (const auto& c : legal) {
        legal_indices.push_back(card_to_index(c));
    }
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int chosen_idx = 0;
    
    const int strategy_choice_i = select_by_strategy(
        legal, hand, current_trick, lead_suit, trump_suit, partner
    );
    const int strategy_idx = legal_indices[strategy_choice_i];

    if (is_best_response) {
        if (dist(rng) < epsilon) {
            std::uniform_int_distribution<int> int_dist(0, legal.size() - 1);
            chosen_idx = legal_indices[int_dist(rng)];
        } else {
            // Use Q-network + basic strategy prior (blackjack basic-strategy analogue).
            std::lock_guard<std::mutex> lk(g_hokm_torch_mutex);
            torch::NoGradGuard no_grad;
            auto state_tensor = torch::tensor(get_state(), torch::kFloat32).to(learner->device);
            auto q_vals = learner->q_network->q_values_at_indices(state_tensor, legal_indices);
            auto strategy_bias = torch::zeros_like(q_vals);
            strategy_bias[strategy_choice_i] = 1.5; // prefer strategy-best unless Q strongly disagrees
            auto blended = q_vals + strategy_bias;
            int max_idx = blended.argmax().item<int>();
            chosen_idx = legal_indices[max_idx];
        }
    } else {
        // Use Policy network + strategy temperature shaping.
        std::lock_guard<std::mutex> lk(g_hokm_torch_mutex);
        torch::NoGradGuard no_grad;
        auto state_tensor = torch::tensor(get_state(), torch::kFloat32).to(learner->device);
        auto logits = learner->policy_network->logits_at_indices(state_tensor, legal_indices);
        logits[strategy_choice_i] += 1.2;
        auto probs = torch::softmax(logits, 0);

        // Sample from distribution
        std::vector<double> probs_vec(probs.data_ptr<float>(), probs.data_ptr<float>() + probs.numel());
        std::discrete_distribution<int> d(probs_vec.begin(), probs_vec.end());
        chosen_idx = legal_indices[d(rng)];
    }
    
    if (std::find(legal_indices.begin(), legal_indices.end(), chosen_idx) == legal_indices.end()) {
        chosen_idx = strategy_idx;
    }
    Card chosen_card = index_to_card(chosen_idx);
    last_strategy_reason = explain_basic_strategy(
        chosen_card, hand, current_trick, lead_suit, trump_suit, partner
    );
    return {chosen_card, chosen_idx};
}

double EnhancedPlayer::evaluate_play(const Card& card, const std::string& lead_suit, int round_num) {
    (void)round_num;
    std::vector<Card> legal = legal_cards_from_hand(hand, lead_suit);
    double s = basic_strategy_score(
        card, legal, hand, current_trick, lead_suit, trump_suit, partner
    );
    // Keep reward shaping bounded and stable.
    if (s > 12.0) s = 12.0;
    if (s < -12.0) s = -12.0;
    return s / 12.0;
}

void EnhancedPlayer::store_experience(const std::vector<float>& state, int action, double reward, 
                                      const std::vector<float>& next_state, bool done, 
                                      bool rl_eligible, const std::vector<bool>& next_legal_mask) {
    if (!learning_enabled) return;
    Experience exp{state, action, reward, next_state, done, rl_eligible, next_legal_mask};
    learner->push_transition(exp, is_best_response);
}

void EnhancedPlayer::optimize_model() {
    if (!learning_enabled) return;
    learner->optimize_q();
    learner->optimize_policy();
}

double EnhancedPlayer::compute_terminal_reward() {
    if (!tricks_won_ptr) return 0.0;
    int my_team = 0;
    int opp_team = 0;
    for (const auto& kv : *tricks_won_ptr) {
        bool same_team = false;
        for (const auto& tp : team) {
            if (tp && kv.first && tp->name == kv.first->name) {
                same_team = true;
                break;
            }
        }
        if (same_team) my_team += kv.second;
        else opp_team += kv.second;
    }
    const int margin = my_team - opp_team;
    if (my_team >= 7) return 2.0 + margin * 0.1;
    if (opp_team >= 7) return -2.0 + margin * 0.1;
    return margin * 0.05;
}

} // namespace hokm
