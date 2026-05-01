#include "enhanced_player.h"
#include "hokm.h"
#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <functional>
#include <array>

#include <torch/torch.h>

using namespace hokm;

namespace {

std::mt19937 g_rng{std::random_device{}()};

enum class SeatMode { Human, AI };
enum class AiPlayStyle { Heuristic, Random };

class ConsolePlayer : public Player {
public:
    explicit ConsolePlayer(
        const std::string& name,
        SeatMode mode,
        AiPlayStyle ai_style = AiPlayStyle::Heuristic
    )
        : Player(name),
          mode_(mode),
          ai_style_(mode == SeatMode::Human ? AiPlayStyle::Heuristic : ai_style) {}

    AiPlayStyle ai_play_style() const { return ai_style_; }

    std::vector<float> get_state() override {
        return std::vector<float>(STATE_DIM, 0.0f);
    }

    std::pair<Card, int> play_card(const std::string& lead_suit) override {
        if (mode_ == SeatMode::Human) {
            if (hand.empty()) {
                throw std::runtime_error("Human seat has no cards to play");
            }
            Card fallback = hand.front();
            return {fallback, card_to_index(fallback)};
        }

        std::vector<Card> legal = legal_cards(lead_suit);
        if (legal.empty()) {
            throw std::runtime_error("AI has no legal cards");
        }
        if (mode_ == SeatMode::AI && ai_style_ == AiPlayStyle::Random) {
            std::uniform_int_distribution<size_t> dist(0, legal.size() - 1);
            const Card pick = legal[dist(g_rng)];
            return {pick, card_to_index(pick)};
        }
        Card best = legal.front();
        for (const auto& c : legal) {
            const bool c_is_trump = (!trump_suit.empty() && c.suit == trump_suit);
            const bool best_is_trump = (!trump_suit.empty() && best.suit == trump_suit);
            if (c_is_trump && !best_is_trump) {
                best = c;
                continue;
            }
            if (c_is_trump == best_is_trump && c.value > best.value) {
                best = c;
            }
        }
        return {best, card_to_index(best)};
    }

    double evaluate_play(const Card&, const std::string&, int) override { return 0.0; }
    void store_experience(const std::vector<float>&, int, double, const std::vector<float>&, bool, bool, const std::vector<bool>&) override {}
    void optimize_model() override {}
    double compute_terminal_reward() override { return 0.0; }
    void _sync_seats(Hokm*) override {}

    bool is_human() const { return mode_ == SeatMode::Human; }
    std::vector<Card> legal_cards(const std::string& lead_suit) const {
        if (lead_suit.empty()) return hand;
        std::vector<Card> following;
        for (const auto& c : hand) {
            if (c.suit == lead_suit) following.push_back(c);
        }
        return following.empty() ? hand : following;
    }

private:
    SeatMode mode_;
    AiPlayStyle ai_style_;
};

struct SessionState {
    std::string id;
    int human_seat = 0;
    std::vector<std::shared_ptr<Player>> players;
    std::string ai_policy = "heuristic";
    std::string model_id;
    std::shared_ptr<SharedNFSPLearner> session_learner;
    std::shared_ptr<Hokm> game;
    std::vector<std::string> event_log;
};

std::unordered_map<std::string, SessionState> g_sessions;
std::mutex g_sessions_mutex;
std::shared_ptr<SharedNFSPLearner> g_trained_learner;
std::string g_trained_model_dir;
std::string g_active_model_id;
int g_trained_episodes = 0;

struct LossSample {
    int episode = 0;
    double q_loss = 0.0;
    double policy_loss = 0.0;
};

struct TrainingJobState {
    bool running = false;
    int requested_episodes = 0;
    int completed_episodes = 0;
    int eval_interval = 0;
    double epsilon = 0.10;
    double eta = 0.10;
    double latest_q_loss = 0.0;
    double latest_policy_loss = 0.0;
    std::vector<LossSample> loss_history;
    std::string last_result_json;
    std::string last_error;
};

TrainingJobState g_training_job;
std::mutex g_training_mutex;

struct EvalJobState {
    bool running = false;
    std::string model_id;
    int total_games = 0;
    int completed_games = 0;
    std::string last_result_json;
    std::string last_error;
};

EvalJobState g_eval_job;
std::mutex g_eval_mutex;

std::filesystem::path models_root_dir() {
    return std::filesystem::path("..") / "models";
}

bool is_safe_model_id(const std::string& id) {
    if (id.empty() || id.size() > 128) return false;
    for (char c : id) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

std::string make_run_id() {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    return "run_" + std::to_string(ms);
}

std::shared_ptr<SharedNFSPLearner> load_learner_from_model_dir(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir / "q_network.pt")) return nullptr;
    auto learner = std::make_shared<SharedNFSPLearner>();
    if (!learner->load_models(dir.string())) return nullptr;
    return learner;
}

int clamp_int(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

int parse_int_or(const std::string& raw, int fallback) {
    try {
        return std::stoi(raw);
    } catch (...) {
        return fallback;
    }
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

std::string json_number(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

std::string list_models_json() {
    std::ostringstream out;
    out << "[";
    bool first = true;
    try {
        const auto root = models_root_dir();
        if (!std::filesystem::exists(root)) {
            out << "]";
            return out.str();
        }
        std::vector<std::pair<std::filesystem::path, std::filesystem::file_time_type>> dirs;
        for (const auto& ent : std::filesystem::directory_iterator(root)) {
            if (!ent.is_directory()) continue;
            if (!std::filesystem::exists(ent.path() / "q_network.pt")) continue;
            dirs.push_back({ent.path(), ent.last_write_time()});
        }
        std::sort(dirs.begin(), dirs.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
        for (const auto& pr : dirs) {
            if (!first) out << ",";
            first = false;
            const std::string id = pr.first.filename().string();
            out << "{\"id\":\"" << json_escape(id) << "\",\"path\":\"" << json_escape(pr.first.string()) << "\"}";
        }
    } catch (...) {}
    out << "]";
    return out.str();
}

std::string cards_to_json(const std::vector<Card>& cards) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < cards.size(); ++i) {
        if (i) out << ",";
        out << "\"" << json_escape(cards[i].to_string()) << "\"";
    }
    out << "]";
    return out.str();
}

std::string session_to_json(const SessionState& s) {
    auto game = s.game;
    auto human = s.players[s.human_seat];
    auto next = game->get_next_to_play();
    auto legal = game->legal_cards_for_player(human);
    const bool game_over = game->is_hand_over();
    std::string winner = "none";
    if (game_over) {
        winner = game->scores.at(1) >= 7 ? "team1" : "team2";
    }

    std::ostringstream out;
    out << "{";
    out << "\"session_id\":\"" << json_escape(s.id) << "\",";
    out << "\"human_seat\":" << s.human_seat << ",";
    out << "\"ai_policy\":\"" << json_escape(s.ai_policy) << "\",";
    out << "\"model_id\":\"" << json_escape(s.model_id) << "\",";
    out << "\"hakem\":\"" << json_escape(game->hakem ? game->hakem->name : "") << "\",";
    out << "\"trump_suit\":\"" << json_escape(game->trump_suit) << "\",";
    out << "\"lead_suit\":\"" << json_escape(game->lead_suit) << "\",";
    out << "\"round\":" << game->round_count << ",";
    out << "\"scores\":{\"team1\":" << game->scores.at(1) << ",\"team2\":" << game->scores.at(2) << "},";
    out << "\"next_player\":\"" << json_escape(next ? next->name : "") << "\",";
    out << "\"is_human_turn\":" << ((next && next->name == human->name) ? "true" : "false") << ",";
    out << "\"game_over\":" << (game_over ? "true" : "false") << ",";
    out << "\"winner\":\"" << winner << "\",";
    out << "\"human_hand\":" << cards_to_json(human->hand) << ",";
    out << "\"legal_cards\":" << cards_to_json(legal) << ",";
    out << "\"players\":[";
    for (size_t i = 0; i < s.players.size(); ++i) {
        if (i) out << ",";
        auto p = s.players[i];
        auto p_legal = game->legal_cards_for_player(p);
        const bool is_human = (static_cast<int>(i) == s.human_seat);
        const bool is_next = (next && next->name == p->name);
        const bool is_team1 = (i == 0 || i == 2);
        out << "{";
        out << "\"seat\":" << i << ",";
        out << "\"name\":\"" << json_escape(p->name) << "\",";
        out << "\"team\":\"" << (is_team1 ? "team1" : "team2") << "\",";
        out << "\"is_human\":" << (is_human ? "true" : "false") << ",";
        out << "\"is_next\":" << (is_next ? "true" : "false") << ",";
        out << "\"hand\":" << cards_to_json(p->hand) << ",";
        out << "\"legal_cards\":" << cards_to_json(p_legal);
        out << "}";
    }
    out << "],";
    out << "\"current_trick\":[";
    for (size_t i = 0; i < game->current_trick.size(); ++i) {
        if (i) out << ",";
        out << "{\"player\":\"" << json_escape(game->current_trick[i].first->name)
            << "\",\"card\":\"" << json_escape(game->current_trick[i].second.to_string()) << "\"}";
    }
    out << "],";
    out << "\"log\":[";
    for (size_t i = 0; i < s.event_log.size(); ++i) {
        if (i) out << ",";
        out << "\"" << json_escape(s.event_log[i]) << "\"";
    }
    out << "]";
    out << "}";
    return out.str();
}

std::string make_session_id() {
    std::uniform_int_distribution<int> d(100000, 999999);
    return "sess-" + std::to_string(d(g_rng));
}

std::string step_one_ai(SessionState& s) {
    auto game = s.game;
    auto next = game->get_next_to_play();
    if (!next) return "no_next_player";
    if (next->name == s.players[s.human_seat]->name) return "human_turn";

    auto [card, _] = next->play_card(game->lead_suit);
    std::string err = game->apply_play(next, card);
    if (!err.empty()) return err;

    std::string reason = "heuristic-default";
    if (auto ep = std::dynamic_pointer_cast<EnhancedPlayer>(next)) {
        reason = ep->last_strategy_reason;
    } else if (auto cp = std::dynamic_pointer_cast<ConsolePlayer>(next)) {
        reason = (cp->ai_play_style() == AiPlayStyle::Random) ? "random-ai" : "console-heuristic";
    }
    s.event_log.push_back(next->name + " played " + card.to_string() + " | reason: " + reason);
    auto winner = game->resolve_trick_if_complete();
    if (winner) {
        s.event_log.push_back(
            "Trick winner: " + winner->name + " | score " +
            std::to_string(game->scores.at(1)) + "-" + std::to_string(game->scores.at(2))
        );
    }
    if (game->is_hand_over()) {
        s.event_log.push_back("Hand finished.");
    }
    return "ok";
}

struct TrainingPoint {
    int episode = 0;
    double team1_win_rate = 0.0;
    double team2_win_rate = 0.0;
    double avg_margin = 0.0;
    double benchmark_win_rate = 0.0;
    double q_loss = 0.0;
    double policy_loss = 0.0;
};

enum class EvalSeatPolicy { Random, Heuristic, Trained };

EvalSeatPolicy parse_eval_seat_policy(const std::string& s) {
    if (s == "random") return EvalSeatPolicy::Random;
    if (s == "trained") return EvalSeatPolicy::Trained;
    return EvalSeatPolicy::Heuristic;
}

const char* eval_seat_policy_name(EvalSeatPolicy p) {
    switch (p) {
        case EvalSeatPolicy::Random:
            return "random";
        case EvalSeatPolicy::Trained:
            return "trained";
        default:
            return "heuristic";
    }
}

std::shared_ptr<Player> make_eval_player(
    int seat_index,
    EvalSeatPolicy pol,
    const std::shared_ptr<SharedNFSPLearner>& learner
) {
    const std::string name = "Player " + std::to_string(seat_index + 1);
    switch (pol) {
        case EvalSeatPolicy::Random:
            return std::make_shared<ConsolePlayer>(name, SeatMode::AI, AiPlayStyle::Random);
        case EvalSeatPolicy::Heuristic:
            return std::make_shared<ConsolePlayer>(name, SeatMode::AI, AiPlayStyle::Heuristic);
        case EvalSeatPolicy::Trained:
            if (!learner) {
                throw std::runtime_error("eval: trained policy requires a loaded NFSP checkpoint");
            }
            {
                auto p = std::make_shared<EnhancedPlayer>(name, learner, 0.0, 1.0, "mixed");
                p->learning_enabled = false;
                return p;
            }
    }
    return std::make_shared<ConsolePlayer>(name, SeatMode::AI, AiPlayStyle::Heuristic);
}

void run_evaluation_with_seats(
    const std::array<EvalSeatPolicy, 4>& seats,
    const std::shared_ptr<SharedNFSPLearner>& learner,
    int games,
    int* team1_wins,
    int* team2_wins,
    const std::function<void(int completed, int total)>& on_progress = nullptr
) {
    *team1_wins = 0;
    *team2_wins = 0;
    for (int g = 0; g < games; ++g) {
        std::vector<std::shared_ptr<Player>> players(4);
        for (int i = 0; i < 4; ++i) {
            players[i] = make_eval_player(i, seats[i], learner);
        }
        Hokm eval_game(players, "", true);
        eval_game.play_game(false);
        if (eval_game.scores[1] >= 7) {
            (*team1_wins)++;
        } else if (eval_game.scores[2] >= 7) {
            (*team2_wins)++;
        }
        if (on_progress && ((g + 1) % 3 == 0 || (g + 1) == games)) {
            on_progress(g + 1, games);
        }
    }
}

double evaluate_trained_vs_heuristic(
    const std::shared_ptr<SharedNFSPLearner>& learner,
    int games,
    const std::function<void(int completed, int total)>& on_progress = nullptr
) {
    int trained_side_wins = 0;
    for (int g = 0; g < games; ++g) {
        const bool trained_on_team1 = (g % 2 == 0);
        std::vector<std::shared_ptr<Player>> players(4);
        for (int i = 0; i < 4; ++i) {
            const bool seat_team1 = (i == 0 || i == 2);
            const bool use_trained = trained_on_team1 ? seat_team1 : !seat_team1;
            const EvalSeatPolicy pol = use_trained ? EvalSeatPolicy::Trained : EvalSeatPolicy::Heuristic;
            players[i] = make_eval_player(i, pol, learner);
        }
        Hokm eval_game(players, "", true);
        eval_game.play_game(false);
        const bool team1_won = eval_game.scores[1] >= 7;
        const bool trained_won = trained_on_team1 ? team1_won : !team1_won;
        if (trained_won) trained_side_wins++;
        if (on_progress && ((g + 1) % 3 == 0 || (g + 1) == games)) {
            on_progress(g + 1, games);
        }
    }
    return (games > 0) ? static_cast<double>(trained_side_wins) / games : 0.0;
}

std::string run_training_report_json(
    int episodes,
    int eval_interval,
    double epsilon,
    double eta,
    const std::function<void(int, int, const std::shared_ptr<SharedNFSPLearner>&)>& on_progress = nullptr
) {
    const std::string run_id = make_run_id();
    auto learner = std::make_shared<SharedNFSPLearner>();
    std::vector<std::shared_ptr<Player>> players = {
        std::make_shared<EnhancedPlayer>("Player 1", learner, epsilon, eta, "mixed"),
        std::make_shared<EnhancedPlayer>("Player 2", learner, epsilon, eta, "mixed"),
        std::make_shared<EnhancedPlayer>("Player 3", learner, epsilon, eta, "mixed"),
        std::make_shared<EnhancedPlayer>("Player 4", learner, epsilon, eta, "mixed")
    };
    Hokm game(players, "", true);

    int t1_total = 0;
    int t2_total = 0;
    std::vector<TrainingPoint> points;
    int chunk_t1 = 0;
    int chunk_t2 = 0;
    double chunk_margin_sum = 0.0;
    int chunk_n = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= episodes; ++i) {
        const double frac = static_cast<double>(i) / std::max(1, episodes);
        const double scheduled_epsilon = std::max(0.02, epsilon * (1.0 - 0.85 * frac));
        for (auto& p : players) {
            auto ep = std::dynamic_pointer_cast<EnhancedPlayer>(p);
            if (ep) ep->epsilon = scheduled_epsilon;
        }
        game.play_game(false);
        if (on_progress && (i % 50 == 0 || i == episodes)) {
            on_progress(i, episodes, learner);
        }
        int margin = game.scores[1] - game.scores[2];
        if (game.scores[1] >= 7) {
            t1_total++;
            chunk_t1++;
        }
        if (game.scores[2] >= 7) {
            t2_total++;
            chunk_t2++;
        }
        chunk_margin_sum += static_cast<double>(margin);
        chunk_n++;

        if (i % eval_interval == 0 || i == episodes) {
            TrainingPoint p;
            p.episode = i;
            p.team1_win_rate = (chunk_n > 0) ? (static_cast<double>(chunk_t1) / chunk_n) : 0.0;
            p.team2_win_rate = (chunk_n > 0) ? (static_cast<double>(chunk_t2) / chunk_n) : 0.0;
            p.avg_margin = (chunk_n > 0) ? (chunk_margin_sum / chunk_n) : 0.0;
            p.benchmark_win_rate = evaluate_trained_vs_heuristic(learner, 80);
            p.q_loss = learner->latest_q_loss;
            p.policy_loss = learner->latest_policy_loss;
            points.push_back(p);
            chunk_t1 = chunk_t2 = 0;
            chunk_margin_sum = 0.0;
            chunk_n = 0;
        }
    }
    auto t1_time = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1_time - t0).count();

    std::ostringstream out;
    out << "{";
    out << "\"episodes\":" << episodes << ",";
    out << "\"run_id\":\"" << json_escape(run_id) << "\",";
    out << "\"eval_interval\":" << eval_interval << ",";
    out << "\"params\":{\"epsilon\":" << json_number(epsilon) << ",\"eta\":" << json_number(eta) << "},";
    out << "\"team1_wins\":" << t1_total << ",";
    out << "\"team2_wins\":" << t2_total << ",";
    out << "\"aborted_games\":" << game.aborted_games << ",";
    out << "\"elapsed_seconds\":" << json_number(secs) << ",";
    out << "\"latest_q_loss\":" << json_number(learner->latest_q_loss) << ",";
    out << "\"latest_policy_loss\":" << json_number(learner->latest_policy_loss) << ",";
    out << "\"games_per_second\":" << json_number(secs > 0 ? static_cast<double>(episodes) / secs : 0.0) << ",";
    out << "\"approach\":\"hybrid_basic_strategy_plus_nfsp_with_throttled_updates\",";
    out << "\"final_benchmark_win_rate_vs_heuristic\":"
        << json_number(points.empty() ? 0.0 : points.back().benchmark_win_rate) << ",";
    out << "\"points\":[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i) out << ",";
        out << "{";
        out << "\"episode\":" << points[i].episode << ",";
        out << "\"team1_win_rate\":" << json_number(points[i].team1_win_rate) << ",";
        out << "\"team2_win_rate\":" << json_number(points[i].team2_win_rate) << ",";
        out << "\"avg_margin\":" << json_number(points[i].avg_margin) << ",";
        out << "\"benchmark_win_rate\":" << json_number(points[i].benchmark_win_rate) << ",";
        out << "\"q_loss\":" << json_number(points[i].q_loss) << ",";
        out << "\"policy_loss\":" << json_number(points[i].policy_loss);
        out << "}";
    }
    out << "]";
    out << "}";

    try {
        const auto root = models_root_dir();
        std::filesystem::create_directories(root);
        const auto run_dir = root / run_id;
        const auto latest_dir = root / "latest";
        std::filesystem::create_directories(run_dir);
        std::filesystem::create_directories(latest_dir);
        learner->save_models(run_dir.string());
        learner->save_models(latest_dir.string());
        g_trained_model_dir = run_dir.string();
        g_active_model_id = run_id;
    } catch (...) {
        // Still return metrics; disk save may fail in restricted environments.
    }
    g_trained_learner = learner;
    g_trained_episodes = episodes;
    return out.str();
}

} // namespace

int main() {
    try {
        torch::set_num_threads(1);
        at::set_num_interop_threads(1);
    } catch (...) {
        // Older LibTorch builds may not expose these; continue anyway.
    }

    httplib::Server svr;
    std::cout << "Starting Hokm C++ Web Server on http://localhost:8080..." << std::endl;
    svr.set_mount_point("/static", "../static");

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        const std::string page = R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Hokm Dev Console</title>
  <style>
    :root {
      --bg0: #070a12;
      --bg1: #0c1222;
      --surface: rgba(18, 24, 42, 0.72);
      --surface2: rgba(12, 18, 34, 0.95);
      --border: rgba(99, 115, 148, 0.35);
      --text: #f1f5f9;
      --muted: #94a3b8;
      --accent: #34d399;
      --accent2: #22d3ee;
      --danger: #fb7185;
      --radius: 14px;
      --shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
      --font: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--font);
      color: var(--text);
      background:
        radial-gradient(1200px 600px at 10% -10%, rgba(52, 211, 153, 0.12), transparent 55%),
        radial-gradient(900px 500px at 100% 20%, rgba(34, 211, 238, 0.08), transparent 50%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }
    .wrap { max-width: 1420px; margin: 0 auto; padding: 28px 22px 48px; }
    .hero {
      margin-bottom: 8px;
    }
    .hero h1 {
      margin: 0 0 6px 0;
      font-size: clamp(1.5rem, 3vw, 1.85rem);
      font-weight: 650;
      letter-spacing: -0.02em;
    }
    .grid { display: grid; grid-template-columns: minmax(280px, 380px) 1fr; gap: 20px; }
    @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
    .tabs {
      display: inline-flex;
      gap: 4px;
      padding: 4px;
      margin-top: 20px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 999px;
      box-shadow: var(--shadow);
    }
    .tab {
      border: none;
      background: transparent;
      color: var(--muted);
      padding: 10px 20px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      margin: 0;
      transition: color 0.15s, background 0.15s;
    }
    .tab:hover { color: var(--text); background: rgba(255,255,255,0.05); }
    .tab.active {
      color: var(--bg0);
      background: linear-gradient(135deg, var(--accent), #2dd4bf);
      box-shadow: 0 2px 12px rgba(52, 211, 153, 0.35);
    }
    .tab-panel { margin-top: 22px; }
    .tab-panel.hidden { display: none; }
    .lab-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
    @media (max-width: 1100px) { .lab-grid { grid-template-columns: 1fr; } }
    .chart-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 700px) { .chart-row { grid-template-columns: 1fr; } }
    .loss-readout { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; font-size: 13px; }
    .loss-readout > div {
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 10px 12px;
    }
    .loss-readout strong { font-variant-numeric: tabular-nums; color: var(--accent2); }
    pre.eval { max-height: 200px; }
    .panel {
      background: var(--surface);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 18px 20px;
      box-shadow: var(--shadow);
    }
    h2 {
      margin: 0 0 14px 0;
      font-size: 0.95rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    .muted { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .btn-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    button {
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.06);
      color: var(--text);
      border-radius: 10px;
      padding: 9px 14px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      margin: 0;
      transition: background 0.15s, border-color 0.15s, transform 0.1s;
    }
    button:hover:not(:disabled) {
      background: rgba(255,255,255,0.1);
      border-color: rgba(148, 163, 184, 0.45);
    }
    button:active:not(:disabled) { transform: scale(0.98); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.primary {
      border: none;
      background: linear-gradient(135deg, var(--accent), #14b8a6);
      color: #042f2e;
      font-weight: 600;
    }
    button.primary:hover:not(:disabled) {
      filter: brightness(1.06);
      box-shadow: 0 4px 20px rgba(52, 211, 153, 0.3);
    }
    .card-btn { display: inline-block; margin: 4px; }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .row label { min-width: 7.5rem; font-size: 12px; color: var(--muted); }
    input, select {
      background: rgba(0,0,0,0.35);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 11px;
      font-size: 13px;
    }
    input:focus, select:focus, button:focus-visible {
      outline: 2px solid rgba(52, 211, 153, 0.45);
      outline-offset: 1px;
    }
    pre {
      background: rgba(0,0,0,0.35);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px;
      max-height: 300px;
      overflow: auto;
      white-space: pre-wrap;
      font-size: 12px;
      line-height: 1.45;
    }
    .progress {
      margin-top: 12px;
      height: 8px;
      background: rgba(0,0,0,0.35);
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .progress > div {
      height: 100%;
      width: 0%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--accent), var(--accent2));
      transition: width 0.25s ease;
    }
    .kpi { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin: 12px 0; }
    @media (max-width: 640px) { .kpi { grid-template-columns: repeat(2, 1fr); } }
    .kpi > div {
      background: rgba(0,0,0,0.25);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
      font-size: 13px;
    }
    .table-area {
      margin-top: 12px;
      background: radial-gradient(ellipse at center, #15803d 0%, #0f5132 45%, #052e16 100%);
      border: 1px solid rgba(52, 211, 153, 0.25);
      border-radius: var(--radius);
      padding: 14px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .table-grid { display: grid; grid-template-columns: 1fr 200px 1fr; grid-template-rows: auto 190px auto; gap: 12px; align-items: center; }
    .seat {
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 12px;
      padding: 10px;
      min-height: 110px;
    }
    .seat.next { border-color: #fbbf24; box-shadow: 0 0 0 1px rgba(251, 191, 36, 0.5) inset; }
    .seat h4 { margin: 0 0 6px 0; font-size: 12px; font-weight: 600; }
    .seat.north { grid-column: 1 / 4; grid-row: 1; }
    .seat.west { grid-column: 1; grid-row: 2; }
    .seat.east { grid-column: 3; grid-row: 2; }
    .seat.south { grid-column: 1 / 4; grid-row: 3; }
    .cards { display: flex; flex-wrap: wrap; gap: 6px; }
    .playing-card {
      width: 42px; height: 60px;
      border-radius: 8px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: rgba(7, 10, 18, 0.9);
      object-fit: cover;
    }
    .playing-card.legal {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(52, 211, 153, 0.35);
      cursor: pointer;
    }
    .playing-card.played { width: 52px; height: 74px; }
    .trick-center {
      grid-column: 2; grid-row: 2;
      background: rgba(7, 10, 18, 0.75);
      border: 1px dashed rgba(148, 163, 184, 0.35);
      border-radius: 12px;
      padding: 10px;
      min-height: 160px;
    }
    .trick-pile { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .trick-item { text-align: center; font-size: 11px; color: #cbd5e1; }
    .help { font-size: 12px; color: var(--muted); margin-top: 8px; line-height: 1.5; }
    .help code {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 6px;
      background: rgba(0,0,0,0.35);
      border: 1px solid var(--border);
    }
    canvas {
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .eval-seat-grid {
      display: grid;
      gap: 8px;
      margin: 12px 0;
      padding: 12px;
      background: rgba(0,0,0,0.2);
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .eval-seat-grid .row { margin: 0; }
    .eval-seat-grid .row label { min-width: 5.5rem; }
  </style>
</head>
<body>
  <div class="wrap">
    <header class="hero">
      <h1>Hokm Lab</h1>
      <div class="muted">Play against NFSP agents, train with live loss curves, pick checkpoints, and benchmark vs heuristic without blocking the UI.</div>
    </header>
    <div class="tabs">
      <button type="button" class="tab active" data-tab="play" onclick="showTab('play')">Play</button>
      <button type="button" class="tab" data-tab="lab" onclick="showTab('lab')">Train &amp; evaluate</button>
    </div>

    <div id="tab-play" class="tab-panel">
    <div class="grid">
      <div class="panel">
        <h2>Session</h2>
        <div class="row">
          <label>Human seat</label>
          <select id="humanSeat">
            <option value="0">Player 1 (Team 1)</option>
            <option value="1">Player 2 (Team 2)</option>
            <option value="2">Player 3 (Team 1)</option>
            <option value="3">Player 4 (Team 2)</option>
          </select>
        </div>
        <div class="row">
          <label>AI policy</label>
          <select id="aiPolicy" onchange="syncCheckpointUi()">
            <option value="heuristic">Heuristic Basic Strategy</option>
            <option value="trained">Trained NFSP Model</option>
          </select>
        </div>
        <div class="row">
          <label>NFSP checkpoint</label>
          <select id="checkpointPlay" disabled title="Pick a saved run folder under models/, or leave default for in-memory active weights">
            <option value="">— Active (memory / latest) —</option>
          </select>
        </div>
        <div class="help">NFSP opponents use the checkpoint you select, or in-memory weights from the last run / <strong>Set active</strong> in the lab.</div>
        <div class="btn-row" style="margin-top:12px;">
          <button type="button" class="primary" onclick="newSession()">New session</button>
          <button type="button" onclick="refreshState()">Refresh</button>
          <button type="button" onclick="stepAI()">Step AI</button>
          <button type="button" onclick="autoPlay()">Auto play</button>
          <button type="button" onclick="refreshModelLists()">Refresh checkpoints</button>
        </div>
        <div class="muted" id="sessionMeta">No active session.</div>
      </div>

      <div>
        <div class="panel">
          <h2>Live Game State</h2>
          <div class="kpi" id="kpis"></div>
          <div class="help">All players are open-hand in development mode for debugging.</div>
          <div class="table-area">
            <div id="tableView" class="table-grid"></div>
          </div>
          <div style="margin-top:10px;"><strong>Your Legal Cards (click to play)</strong></div>
          <div id="hand"></div>
        </div>
        <div class="panel" style="margin-top:16px;">
          <h2>Basic Strategy Report</h2>
          <pre id="strategyReport">No strategy events yet.</pre>
          <h2>Developer Logs</h2>
          <pre id="logs">No logs yet.</pre>
          <h2 style="margin-top:12px;">Raw JSON Snapshot</h2>
          <pre id="raw">N/A</pre>
        </div>
      </div>
    </div>
    </div>

    <div id="tab-lab" class="tab-panel hidden">
    <div class="lab-grid">
      <div class="panel">
        <h2>Training</h2>
        <div class="row">
          <label>Episodes</label><input id="episodes" type="number" min="10" max="100000" value="500"/>
        </div>
        <div class="row">
          <label>Eval interval</label><input id="evalInterval" type="number" min="5" max="20000" value="1000"/>
        </div>
        <div class="row">
          <label>Epsilon</label><input id="epsilon" type="number" step="0.01" min="0" max="1" value="0.10"/>
        </div>
        <div class="row">
          <label>Eta</label><input id="eta" type="number" step="0.01" min="0" max="1" value="0.10"/>
        </div>
        <div class="btn-row">
          <button type="button" class="primary" onclick="train()">Run training</button>
          <button type="button" onclick="train10k()">10k deep run</button>
        </div>
        <div class="progress" style="margin-top:14px;"><div id="trainProgressBar"></div></div>
        <div id="trainProgressLabel" class="muted" style="margin-top:6px;">0% (0/0)</div>
        <div class="loss-readout">
          <div><span class="muted">Q loss (latest)</span><br/><strong id="lossQ">—</strong></div>
          <div><span class="muted">Policy loss (latest)</span><br/><strong id="lossPi">—</strong></div>
        </div>
        <pre id="trainingOut">Not started.</pre>
        <div class="chart-row">
          <div>
            <div class="muted" style="margin-bottom:4px;">Win rates &amp; benchmark</div>
            <canvas id="trainChart" width="420" height="200" style="width:100%;height:200px;background:rgba(0,0,0,0.35);"></canvas>
          </div>
          <div>
            <div class="muted" style="margin-bottom:4px;">Losses (live while training)</div>
            <canvas id="lossChart" width="420" height="200" style="width:100%;height:200px;background:rgba(0,0,0,0.35);"></canvas>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2>Evaluation</h2>
        <p class="muted" style="margin-top:0;">Run repeated full games with independent controllers per seat: random legal play, heuristic, or trained NFSP. Team 1 is seats P1 and P3; team 2 is P2 and P4.</p>
        <div class="eval-seat-grid">
          <div class="muted" style="font-size:12px;">Algorithm per player</div>
          <div class="row">
            <label>P1 · T1</label>
            <select id="evalSeat0" onchange="syncEvalTrainedHint()">
              <option value="random">Random</option>
              <option value="heuristic">Heuristic</option>
              <option value="trained" selected>Trained NFSP</option>
            </select>
          </div>
          <div class="row">
            <label>P2 · T2</label>
            <select id="evalSeat1" onchange="syncEvalTrainedHint()">
              <option value="random">Random</option>
              <option value="heuristic" selected>Heuristic</option>
              <option value="trained">Trained NFSP</option>
            </select>
          </div>
          <div class="row">
            <label>P3 · T1</label>
            <select id="evalSeat2" onchange="syncEvalTrainedHint()">
              <option value="random">Random</option>
              <option value="heuristic">Heuristic</option>
              <option value="trained" selected>Trained NFSP</option>
            </select>
          </div>
          <div class="row">
            <label>P4 · T2</label>
            <select id="evalSeat3" onchange="syncEvalTrainedHint()">
              <option value="random">Random</option>
              <option value="heuristic" selected>Heuristic</option>
              <option value="trained">Trained NFSP</option>
            </select>
          </div>
        </div>
        <div class="row">
          <label>NFSP checkpoint</label>
          <select id="evalModel" style="flex:1;min-width:0;">
            <option value="">— Active / latest (memory) —</option>
          </select>
        </div>
        <div id="evalCheckpointHint" class="help">Trained NFSP seats need weights: choose a saved run above, or train / <strong>Set active</strong> first.</div>
        <div class="row">
          <label>Games</label><input id="evalGames" type="number" min="20" max="5000" value="200"/>
        </div>
        <div class="btn-row">
          <button type="button" class="primary" id="btnBenchmark" onclick="runBenchmark()">Run benchmark</button>
          <button type="button" onclick="activateCheckpoint()">Set active</button>
          <button type="button" onclick="refreshModelLists()">Refresh list</button>
        </div>
        <div class="progress" style="margin-top:12px;"><div id="evalProgressBar"></div></div>
        <div id="evalProgressLabel" class="muted" style="margin-top:6px;">Idle</div>
        <pre id="evalResult" class="eval muted">No benchmark yet. Runs in the background; progress updates here.</pre>
        <div class="help">Training saves each run under <code>cpp_version/models/&lt;run_id&gt;</code> and updates <code>models/latest</code>. Activating loads that folder into memory for play sessions that use the default checkpoint.</div>
      </div>
    </div>
    </div>
  </div>
<script>
let sessionId = "";
let state = null;

function showTab(name) {
  document.querySelectorAll(".tab").forEach(t => {
    t.classList.toggle("active", t.getAttribute("data-tab") === name);
  });
  document.getElementById("tab-play").classList.toggle("hidden", name !== "play");
  document.getElementById("tab-lab").classList.toggle("hidden", name !== "lab");
}

function syncCheckpointUi() {
  const trained = document.getElementById("aiPolicy").value === "trained";
  document.getElementById("checkpointPlay").disabled = !trained;
}

function syncEvalTrainedHint() {
  let need = false;
  for (let i = 0; i < 4; i++) {
    if (document.getElementById("evalSeat" + i).value === "trained") need = true;
  }
  const hint = document.getElementById("evalCheckpointHint");
  if (hint) hint.style.display = need ? "block" : "none";
}

function evalSeatQuery() {
  let q = "";
  for (let i = 0; i < 4; i++) {
    q += `&seat${i}=${encodeURIComponent(document.getElementById("evalSeat" + i).value)}`;
  }
  return q;
}

async function refreshModelLists() {
  try {
    const data = await api("/api/models/list");
    const models = data.models || [];
    const selects = [
      document.getElementById("checkpointPlay"),
      document.getElementById("evalModel")
    ];
    for (const sel of selects) {
      const prev = sel.value;
      const isPlay = sel.id === "checkpointPlay";
      sel.innerHTML = isPlay
        ? "<option value=\"\">— Active (memory / latest) —</option>"
        : "<option value=\"\">— Active / latest (memory) —</option>";
      models.forEach(m => {
        const o = document.createElement("option");
        o.value = m.id;
        o.textContent = m.id;
        sel.appendChild(o);
      });
      if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
    }
  } catch (e) {
    console.warn(e);
  }
}

async function api(path) {
  let r;
  try {
    r = await fetch(path);
  } catch (e) {
    throw new Error(`Network error (${path}): ${e && e.message ? e.message : e}`);
  }
  const text = await r.text();
  if (!r.ok) {
    throw new Error(`HTTP ${r.status} (${path}): ${text.slice(0, 500)}`);
  }
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error(`Invalid JSON from ${path}: ${e.message}. Snippet: ${text.slice(0, 240)}`);
  }
}

function updateView() {
  if (!state) return;
  let sm = `Session ${state.session_id} | next: ${state.next_player} | human turn: ${state.is_human_turn}`;
  if (state.model_id) sm += ` | NFSP: ${state.model_id}`;
  document.getElementById("sessionMeta").textContent = sm;

  document.getElementById("kpis").innerHTML = `
    <div><div class="muted">Trump</div><div>${state.trump_suit}</div></div>
    <div><div class="muted">Lead Suit</div><div>${state.lead_suit || "-"}</div></div>
    <div><div class="muted">Score</div><div>T1 ${state.scores.team1} - T2 ${state.scores.team2}</div></div>
    <div><div class="muted">Round</div><div>${state.round}</div></div>
  `;

  const cardImageSrc = (cardName) => {
    const normalized = (cardName || "")
      .toLowerCase()
      .replaceAll(" of ", "_of_")
      .replaceAll(" ", "_");
    return `/static/cards/${normalized}.png`;
  };

  const table = document.getElementById("tableView");
  const players = state.players || [];
  const bySeat = {};
  players.forEach(p => { bySeat[p.seat] = p; });
  const seatClass = {0: "south", 1: "west", 2: "north", 3: "east"};

  const seatHtml = [0,1,2,3].map(i => {
    const p = bySeat[i];
    if (!p) return "";
    const cards = (p.hand || []).map(c => {
      const legal = (p.legal_cards || []).includes(c);
      const clickable = p.is_human && state.is_human_turn && legal;
      return `<img src="${cardImageSrc(c)}" alt="${c}" title="${c}" class="playing-card ${legal ? "legal" : ""}" ${clickable ? `onclick='playCard(${JSON.stringify(c)})'` : ""} onerror="this.style.opacity='0.35';this.title='Missing image: ${c}'" />`;
    }).join("") || `<span class="muted">No cards</span>`;
    return `
      <div class="seat ${seatClass[i]} ${p.is_next ? "next" : ""}">
        <h4>${p.name} (${p.team}) ${p.is_human ? "· You" : ""} ${p.is_next ? "· Turn" : ""}</h4>
        <div class="cards">${cards}</div>
      </div>`;
  }).join("");

  const trickItems = (state.current_trick || []).map(t => `
    <div class="trick-item">
      <div>${t.player}</div>
      <img src="${cardImageSrc(t.card)}" alt="${t.card}" title="${t.card}" class="playing-card played" onerror="this.style.opacity='0.35'" />
    </div>
  `).join("") || `<div class="muted">No cards on table</div>`;

  table.innerHTML = `
    ${seatHtml}
    <div class="trick-center">
      <div><strong>Current Trick</strong></div>
      <div class="trick-pile">${trickItems}</div>
    </div>
  `;

  document.getElementById("hand").innerHTML = (state.legal_cards || []).map(c =>
    `<button class='card-btn' ${state.is_human_turn ? "" : "disabled"} onclick='playCard(${JSON.stringify(c)})'>${c}</button>`
  ).join("") || "<span class='muted'>No legal cards.</span>";

  document.getElementById("logs").textContent = (state.log || []).join("\n");
  const strategyLines = (state.log || []).filter(x => x.includes("| reason:"));
  document.getElementById("strategyReport").textContent = strategyLines.length
    ? strategyLines.join("\n")
    : "No strategy events yet.";
  document.getElementById("raw").textContent = JSON.stringify(state, null, 2);
}

async function newSession() {
  const humanSeat = Number(document.getElementById("humanSeat").value || 0);
  const aiPolicy = document.getElementById("aiPolicy").value || "heuristic";
  let url = `/api/session/new?human_seat=${humanSeat}&ai_policy=${encodeURIComponent(aiPolicy)}`;
  if (aiPolicy === "trained") {
    const mid = document.getElementById("checkpointPlay").value;
    if (mid) url += `&model_id=${encodeURIComponent(mid)}`;
  }
  const data = await api(url);
  sessionId = data.session_id;
  state = data;
  updateView();
}

async function refreshState() {
  if (!sessionId) return;
  state = await api(`/api/session/state?session_id=${encodeURIComponent(sessionId)}`);
  updateView();
}

async function playCard(card) {
  if (!sessionId) return;
  state = await api(`/api/session/play?session_id=${encodeURIComponent(sessionId)}&card=${encodeURIComponent(card)}`);
  updateView();
}

async function stepAI() {
  if (!sessionId) return;
  state = await api(`/api/session/step_ai?session_id=${encodeURIComponent(sessionId)}`);
  updateView();
}

async function autoPlay() {
  if (!sessionId) return;
  state = await api(`/api/session/auto_play?session_id=${encodeURIComponent(sessionId)}`);
  updateView();
}

async function train() {
  const episodes = Number(document.getElementById("episodes").value || 500);
  const evalInterval = Number(document.getElementById("evalInterval").value || 1000);
  const epsilon = Number(document.getElementById("epsilon").value || 0.1);
  const eta = Number(document.getElementById("eta").value || 0.1);
  const outEl = document.getElementById("trainingOut");
  outEl.textContent = "Starting...";
  try {
    await api(`/api/train/start?episodes=${episodes}&eval_interval=${evalInterval}&epsilon=${epsilon}&eta=${eta}`);
  } catch (e) {
    outEl.textContent = "Could not start training: " + e;
    return;
  }
  await pollTrainingStatus();
}

async function train10k() {
  document.getElementById("episodes").value = 10000;
  document.getElementById("evalInterval").value = 1000;
  await train();
}

async function pollTrainingStatus() {
  const bar = document.getElementById("trainProgressBar");
  const label = document.getElementById("trainProgressLabel");
  const outEl = document.getElementById("trainingOut");
  try {
    while (true) {
      let st;
      try {
        st = await api("/api/train/status");
      } catch (e) {
        outEl.textContent =
          "Stopped hearing from the training server (HokmApp may have quit, crashed, or the network failed).\n" +
          "If it crashed, check the terminal where HokmApp is running for a C++/LibTorch error.\n\n" + e;
        break;
      }
      const total = Math.max(1, st.requested_episodes || 1);
      const done = Math.max(0, st.completed_episodes || 0);
      const pct = Math.max(0, Math.min(100, Math.floor((done / total) * 100)));
      bar.style.width = `${pct}%`;
      label.textContent = `${pct}% (${done}/${total})`;
      const lq = document.getElementById("lossQ");
      const lp = document.getElementById("lossPi");
      const qv = st.latest_q_loss;
      const pv = st.latest_policy_loss;
      lq.textContent =
        qv != null && Number.isFinite(Number(qv)) ? Number(qv).toFixed(6) : "—";
      lp.textContent =
        pv != null && Number.isFinite(Number(pv)) ? Number(pv).toFixed(6) : "—";
      renderLossChart(st.loss_history || []);
      if (!st.running) {
        if (st.error && st.error.length) {
          outEl.textContent = "Training failed: " + st.error;
        } else if (st.has_result) {
          try {
            const pack = await api("/api/train/result");
            const result = pack.result;
            outEl.textContent = JSON.stringify(result, null, 2);
            renderTrainingChart(result.points || []);
            const pts = (result.points || []).map(p => ({
              episode: p.episode,
              q_loss: p.q_loss,
              policy_loss: p.policy_loss
            }));
            renderLossChart(pts);
          } catch (e2) {
            outEl.textContent =
              "Training finished but loading the full result failed.\n\n" + e2;
          }
        } else {
          outEl.textContent = "Training finished with no result payload.";
        }
        refreshModelLists();
        break;
      }
      await new Promise(r => setTimeout(r, 1000));
    }
  } catch (e) {
    outEl.textContent = "Training monitor error: " + e;
  }
}

async function runBenchmark() {
  const id = document.getElementById("evalModel").value;
  const out = document.getElementById("evalResult");
  const bar = document.getElementById("evalProgressBar");
  const label = document.getElementById("evalProgressLabel");
  const btn = document.getElementById("btnBenchmark");
  const games = Number(document.getElementById("evalGames").value || 200);
  out.textContent = "Starting background benchmark…";
  if (btn) btn.disabled = true;
  try {
    let startUrl = `/api/eval/start?games=${games}${evalSeatQuery()}`;
    if (id) startUrl += `&model_id=${encodeURIComponent(id)}`;
    const start = await fetch(startUrl);
    const st0 = await start.json().catch(() => ({}));
    if (!start.ok) {
      out.textContent = JSON.stringify(st0, null, 2);
      return;
    }
    while (true) {
      let st;
      try {
        st = await api("/api/eval/status");
      } catch (e) {
        out.textContent = "Lost connection during evaluation.\n\n" + e;
        break;
      }
      const total = Math.max(1, st.total_games || 1);
      const done = Math.max(0, st.completed_games || 0);
      const pct = Math.max(0, Math.min(100, Math.floor((done / total) * 100)));
      if (bar) bar.style.width = `${pct}%`;
      if (label) {
        label.textContent = st.running
          ? `Evaluating… ${pct}% (${done} / ${total} games)`
          : "Done.";
      }
      if (!st.running) {
        if (st.error && st.error.length) {
          out.textContent = "Error: " + st.error;
        } else if (st.result) {
          try {
            out.textContent = JSON.stringify(st.result, null, 2);
          } catch (e) {
            out.textContent = String(st.result);
          }
        } else {
          out.textContent = "Finished with no result payload.";
        }
        break;
      }
      await new Promise(r => setTimeout(r, 400));
    }
  } catch (e) {
    out.textContent = String(e);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function activateCheckpoint() {
  const id = document.getElementById("evalModel").value;
  const out = document.getElementById("evalResult");
  if (!id) {
    out.textContent = "Select a checkpoint first.";
    return;
  }
  const r = await fetch(`/api/models/activate?model_id=${encodeURIComponent(id)}`);
  const txt = await r.text();
  try {
    out.textContent = JSON.stringify(JSON.parse(txt), null, 2);
  } catch {
    out.textContent = txt;
  }
  refreshModelLists();
}

function renderLossChart(samples) {
  const canvas = document.getElementById("lossChart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(8, 12, 22, 0.92)";
  ctx.fillRect(0, 0, w, h);
  if (!samples || !samples.length) return;
  const pad = 24;
  const maxEp = samples[samples.length - 1].episode || 1;
  let maxL = 1e-9;
  samples.forEach(s => {
    maxL = Math.max(maxL, Math.abs(s.q_loss || 0), Math.abs(s.policy_loss || 0));
  });
  const x = ep => pad + (ep / maxEp) * (w - 2 * pad);
  const y = v => h - pad - (Math.min(Math.abs(v), maxL) / maxL) * (h - 2 * pad);
  ctx.strokeStyle = "#334155";
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.stroke();
  ctx.strokeStyle = "#c084fc";
  ctx.beginPath();
  samples.forEach((s, i) => {
    const px = x(s.episode), py = y(s.q_loss || 0);
    if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.strokeStyle = "#f472b6";
  ctx.beginPath();
  samples.forEach((s, i) => {
    const px = x(s.episode), py = y(s.policy_loss || 0);
    if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px sans-serif";
  ctx.fillText("Violet: Q · Pink: policy", pad, 12);
}

function renderTrainingChart(points) {
  const canvas = document.getElementById("trainChart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(8, 12, 22, 0.92)";
  ctx.fillRect(0, 0, w, h);
  if (!points.length) return;

  const pad = 20;
  const maxX = points[points.length - 1].episode || 1;
  const yMin = -7, yMax = 1;
  const x = (ep) => pad + (ep / maxX) * (w - 2 * pad);
  const yRate = (v) => h - pad - (v * (h - 2 * pad)); // 0..1
  const yMargin = (v) => h - pad - ((v - yMin) / (yMax - yMin)) * (h - 2 * pad);

  ctx.strokeStyle = "#334155";
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.stroke();

  ctx.strokeStyle = "#22c55e";
  ctx.beginPath();
  points.forEach((p, i) => {
    const px = x(p.episode), py = yRate(p.team1_win_rate || 0);
    if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  ctx.strokeStyle = "#ef4444";
  ctx.beginPath();
  points.forEach((p, i) => {
    const px = x(p.episode), py = yRate(p.team2_win_rate || 0);
    if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  ctx.strokeStyle = "#60a5fa";
  ctx.beginPath();
  points.forEach((p, i) => {
    const px = x(p.episode), py = yMargin(p.avg_margin || 0);
    if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  ctx.strokeStyle = "#f59e0b";
  ctx.beginPath();
  points.forEach((p, i) => {
    const px = x(p.episode), py = yRate(p.benchmark_win_rate || 0);
    if (!i) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  ctx.stroke();

  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px sans-serif";
  ctx.fillText("Green: Team1 win-rate", pad, 10);
  ctx.fillText("Red: Team2 win-rate", pad + 110, 10);
  ctx.fillText("Blue: Avg margin", pad + 220, 10);
  ctx.fillText("Orange: Win-rate vs heuristic", pad, 22);
}
syncCheckpointUi();
syncEvalTrainedHint();
refreshModelLists();
</script>
</body>
</html>
)HTML";
        res.set_content(page, "text/html");
    });

    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\",\"service\":\"hokm-cpp-dev-console\"}", "application/json");
    });

    svr.Get("/api/model/status", [](const httplib::Request&, httplib::Response& res) {
        std::ostringstream out;
        out << "{";
        out << "\"trained_available\":" << (g_trained_learner ? "true" : "false") << ",";
        out << "\"trained_episodes\":" << g_trained_episodes << ",";
        out << "\"active_model_id\":\"" << json_escape(g_active_model_id) << "\",";
        out << "\"model_dir\":\"" << json_escape(g_trained_model_dir) << "\"";
        out << "}";
        res.set_content(out.str(), "application/json");
    });

    svr.Get("/api/models/list", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"models\":" + list_models_json() + "}", "application/json");
    });

    svr.Get("/api/models/activate", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("model_id")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing model_id\"}", "application/json");
            return;
        }
        const std::string id = req.get_param_value("model_id");
        if (!is_safe_model_id(id)) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid model_id\"}", "application/json");
            return;
        }
        auto learner = load_learner_from_model_dir(models_root_dir() / id);
        if (!learner) {
            res.status = 404;
            res.set_content("{\"error\":\"checkpoint not found\"}", "application/json");
            return;
        }
        g_trained_learner = learner;
        g_active_model_id = id;
        g_trained_model_dir = (models_root_dir() / id).string();
        std::ostringstream out;
        out << "{\"status\":\"ok\",\"active_model_id\":\"" << json_escape(id) << "\"}";
        res.set_content(out.str(), "application/json");
    });

    svr.Get("/api/eval/start", [](const httplib::Request& req, httplib::Response& res) {
        std::array<EvalSeatPolicy, 4> seats{};
        for (int i = 0; i < 4; ++i) {
            const std::string key = "seat" + std::to_string(i);
            const std::string def = (i == 0 || i == 2) ? "trained" : "heuristic";
            const std::string val = req.has_param(key) ? req.get_param_value(key) : def;
            seats[static_cast<size_t>(i)] = parse_eval_seat_policy(val);
        }
        int games = 200;
        if (req.has_param("games")) {
            games = clamp_int(parse_int_or(req.get_param_value("games"), 200), 1, 5000);
        }
        std::string id;
        if (req.has_param("model_id")) {
            id = req.get_param_value("model_id");
        }
        bool any_trained = false;
        for (EvalSeatPolicy p : seats) {
            if (p == EvalSeatPolicy::Trained) any_trained = true;
        }
        if (any_trained && !id.empty() && !is_safe_model_id(id)) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid model_id\"}", "application/json");
            return;
        }
        {
            std::lock_guard<std::mutex> lk(g_eval_mutex);
            if (g_eval_job.running) {
                res.status = 409;
                res.set_content("{\"error\":\"evaluation already running\"}", "application/json");
                return;
            }
        }
        {
            std::lock_guard<std::mutex> lk(g_training_mutex);
            if (g_training_job.running) {
                res.status = 409;
                res.set_content("{\"error\":\"training in progress\"}", "application/json");
                return;
            }
        }
        std::shared_ptr<SharedNFSPLearner> learner;
        if (any_trained) {
            if (!id.empty()) {
                learner = load_learner_from_model_dir(models_root_dir() / id);
            }
            if (!learner) {
                learner = g_trained_learner;
            }
            if (!learner) {
                res.status = 400;
                res.set_content(
                    "{\"error\":\"trained seats need a checkpoint (dropdown) or in-memory weights from training / Set active\"}",
                    "application/json"
                );
                return;
            }
        }
        const std::string job_model_label = id.empty() ? g_active_model_id : id;

        {
            std::lock_guard<std::mutex> lk(g_eval_mutex);
            g_eval_job.running = true;
            g_eval_job.model_id = job_model_label;
            g_eval_job.total_games = games;
            g_eval_job.completed_games = 0;
            g_eval_job.last_error.clear();
            g_eval_job.last_result_json.clear();
        }

        std::thread([seats, learner, games, id]() {
            try {
                int t1 = 0;
                int t2 = 0;
                run_evaluation_with_seats(
                    seats,
                    learner,
                    games,
                    &t1,
                    &t2,
                    [](int completed, int total) {
                        std::lock_guard<std::mutex> lk(g_eval_mutex);
                        g_eval_job.completed_games = completed;
                        (void)total;
                    }
                );
                std::ostringstream out;
                out << "{";
                out << "\"games\":" << games << ",";
                out << "\"model_id\":\"" << json_escape(id) << "\",";
                out << "\"seats\":[";
                for (int i = 0; i < 4; ++i) {
                    if (i) out << ",";
                    out << "\"" << eval_seat_policy_name(seats[static_cast<size_t>(i)]) << "\"";
                }
                out << "],";
                out << "\"team1_wins\":" << t1 << ",";
                out << "\"team2_wins\":" << t2 << ",";
                out << "\"team1_win_rate\":" << json_number(games > 0 ? static_cast<double>(t1) / games : 0.0) << ",";
                out << "\"team2_win_rate\":" << json_number(games > 0 ? static_cast<double>(t2) / games : 0.0);
                out << "}";
                std::lock_guard<std::mutex> lk(g_eval_mutex);
                g_eval_job.last_result_json = out.str();
                g_eval_job.completed_games = games;
                g_eval_job.running = false;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(g_eval_mutex);
                g_eval_job.last_error = e.what();
                g_eval_job.running = false;
            } catch (...) {
                std::lock_guard<std::mutex> lk(g_eval_mutex);
                g_eval_job.last_error = "benchmark failed";
                g_eval_job.running = false;
            }
        }).detach();

        res.set_content("{\"status\":\"started\"}", "application/json");
    });

    svr.Get("/api/eval/status", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_eval_mutex);
        std::ostringstream out;
        out << "{";
        out << "\"running\":" << (g_eval_job.running ? "true" : "false") << ",";
        out << "\"model_id\":\"" << json_escape(g_eval_job.model_id) << "\",";
        out << "\"total_games\":" << g_eval_job.total_games << ",";
        out << "\"completed_games\":" << g_eval_job.completed_games << ",";
        out << "\"error\":\"" << json_escape(g_eval_job.last_error) << "\",";
        out << "\"result\":";
        if (g_eval_job.last_result_json.empty()) out << "null";
        else out << g_eval_job.last_result_json;
        out << "}";
        res.set_content(out.str(), "application/json");
    });

    svr.Get("/api/session/new", [](const httplib::Request& req, httplib::Response& res) {
        int human_seat = 0;
        std::string ai_policy = "heuristic";
        std::string model_id_param;
        if (req.has_param("human_seat")) {
            human_seat = clamp_int(parse_int_or(req.get_param_value("human_seat"), 0), 0, 3);
        }
        if (req.has_param("ai_policy")) {
            ai_policy = req.get_param_value("ai_policy");
        }
        if (req.has_param("model_id")) {
            model_id_param = req.get_param_value("model_id");
        }
        if (ai_policy != "heuristic" && ai_policy != "trained") ai_policy = "heuristic";

        std::shared_ptr<SharedNFSPLearner> trainer;
        if (ai_policy == "trained") {
            if (!model_id_param.empty() && is_safe_model_id(model_id_param)) {
                trainer = load_learner_from_model_dir(models_root_dir() / model_id_param);
            }
            if (!trainer && g_trained_learner) {
                trainer = g_trained_learner;
            }
            if (!trainer) {
                ai_policy = "heuristic";
            }
        }

        SessionState s;
        s.id = make_session_id();
        s.human_seat = human_seat;
        s.ai_policy = ai_policy;
        if (ai_policy == "trained" && trainer) {
            if (trainer != g_trained_learner) {
                s.session_learner = trainer;
                s.model_id = model_id_param;
            } else {
                s.session_learner.reset();
                s.model_id = g_active_model_id.empty() ? "latest" : g_active_model_id;
            }
        }
        s.players.resize(4);
        for (int i = 0; i < 4; ++i) {
            if (i == human_seat) {
                s.players[i] = std::make_shared<ConsolePlayer>("Player " + std::to_string(i + 1), SeatMode::Human);
            } else if (ai_policy == "trained" && trainer) {
                auto p = std::make_shared<EnhancedPlayer>(
                    "Player " + std::to_string(i + 1), trainer, 0.0, 1.0, "mixed"
                );
                p->learning_enabled = false;
                s.players[i] = p;
            } else {
                s.players[i] = std::make_shared<ConsolePlayer>("Player " + std::to_string(i + 1), SeatMode::AI);
            }
        }
        s.game = std::make_shared<Hokm>(s.players, "", true);
        s.game->start_game();
        s.game->choose_trump_suit();
        s.event_log.push_back("Session created.");
        s.event_log.push_back("Hakem: " + s.game->hakem->name + ", Trump: " + s.game->trump_suit);
        s.event_log.push_back("AI policy: " + s.ai_policy);
        if (ai_policy == "trained" && !s.model_id.empty()) {
            s.event_log.push_back("NFSP checkpoint: " + s.model_id);
        }

        {
            std::lock_guard<std::mutex> lk(g_sessions_mutex);
            g_sessions[s.id] = s;
        }
        res.set_content(session_to_json(s), "application/json");
    });

    svr.Get("/api/session/state", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("session_id")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing session_id\"}", "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(g_sessions_mutex);
        auto it = g_sessions.find(req.get_param_value("session_id"));
        if (it == g_sessions.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"session not found\"}", "application/json");
            return;
        }
        res.set_content(session_to_json(it->second), "application/json");
    });

    svr.Get("/api/session/play", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("session_id") || !req.has_param("card")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing session_id or card\"}", "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(g_sessions_mutex);
        auto it = g_sessions.find(req.get_param_value("session_id"));
        if (it == g_sessions.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"session not found\"}", "application/json");
            return;
        }
        auto& s = it->second;
        auto game = s.game;
        auto human = s.players[s.human_seat];
        auto next = game->get_next_to_play();
        if (!next || next->name != human->name) {
            res.status = 409;
            res.set_content("{\"error\":\"not human turn\"}", "application/json");
            return;
        }

        try {
            Card chosen = Card::from_string(req.get_param_value("card"));
            std::string err = game->apply_play(human, chosen);
            if (!err.empty()) {
                res.status = 400;
                res.set_content("{\"error\":\"" + json_escape(err) + "\"}", "application/json");
                return;
            }
            s.event_log.push_back("Human played " + chosen.to_string());
            auto winner = game->resolve_trick_if_complete();
            if (winner) {
                s.event_log.push_back("Trick winner: " + winner->name);
            }
            res.set_content(session_to_json(s), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content("{\"error\":\"" + json_escape(e.what()) + "\"}", "application/json");
        }
    });

    svr.Get("/api/session/step_ai", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("session_id")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing session_id\"}", "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(g_sessions_mutex);
        auto it = g_sessions.find(req.get_param_value("session_id"));
        if (it == g_sessions.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"session not found\"}", "application/json");
            return;
        }
        auto status = step_one_ai(it->second);
        if (status != "ok" && status != "human_turn") {
            res.status = 400;
            res.set_content("{\"error\":\"" + json_escape(status) + "\"}", "application/json");
            return;
        }
        res.set_content(session_to_json(it->second), "application/json");
    });

    svr.Get("/api/session/auto_play", [](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("session_id")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing session_id\"}", "application/json");
            return;
        }
        std::lock_guard<std::mutex> lk(g_sessions_mutex);
        auto it = g_sessions.find(req.get_param_value("session_id"));
        if (it == g_sessions.end()) {
            res.status = 404;
            res.set_content("{\"error\":\"session not found\"}", "application/json");
            return;
        }
        auto& s = it->second;
        int guard = 400;
        while (!s.game->is_hand_over() && guard-- > 0) {
            auto status = step_one_ai(s);
            if (status == "human_turn") {
                auto human = s.players[s.human_seat];
                auto legal = s.game->legal_cards_for_player(human);
                if (!legal.empty()) {
                    s.game->apply_play(human, legal.front());
                    s.event_log.push_back("Auto-human played " + legal.front().to_string());
                    auto winner = s.game->resolve_trick_if_complete();
                    if (winner) s.event_log.push_back("Trick winner: " + winner->name);
                } else {
                    break;
                }
            } else if (status != "ok") {
                break;
            }
        }
        res.set_content(session_to_json(s), "application/json");
    });

    svr.Get("/api/train", [](const httplib::Request& req, httplib::Response& res) {
        int episodes = 100;
        if (req.has_param("episodes")) {
            episodes = clamp_int(parse_int_or(req.get_param_value("episodes"), 100), 1, 2000);
        }
        res.set_content(run_training_report_json(episodes, std::max(5, episodes / 10), 0.10, 0.10), "application/json");
    });

    svr.Get("/api/train/advanced", [](const httplib::Request& req, httplib::Response& res) {
        int episodes = 500;
        int eval_interval = 1000;
        if (req.has_param("episodes")) episodes = parse_int_or(req.get_param_value("episodes"), 500);
        if (req.has_param("eval_interval")) eval_interval = parse_int_or(req.get_param_value("eval_interval"), 1000);
        episodes = clamp_int(episodes, 10, 100000);
        eval_interval = clamp_int(eval_interval, 5, 20000);
        double epsilon = 0.10;
        double eta = 0.10;
        try {
            if (req.has_param("epsilon")) epsilon = std::stod(req.get_param_value("epsilon"));
            if (req.has_param("eta")) eta = std::stod(req.get_param_value("eta"));
        } catch (...) {
            epsilon = 0.10;
            eta = 0.10;
        }
        epsilon = std::max(0.0, std::min(1.0, epsilon));
        eta = std::max(0.0, std::min(1.0, eta));
        res.set_content(
            run_training_report_json(episodes, eval_interval, epsilon, eta),
            "application/json"
        );
    });

    svr.Get("/api/train/start", [](const httplib::Request& req, httplib::Response& res) {
        int episodes = 500;
        int eval_interval = 1000;
        if (req.has_param("episodes")) episodes = parse_int_or(req.get_param_value("episodes"), 500);
        if (req.has_param("eval_interval")) eval_interval = parse_int_or(req.get_param_value("eval_interval"), 1000);
        episodes = clamp_int(episodes, 10, 100000);
        eval_interval = clamp_int(eval_interval, 5, 20000);
        double epsilon = 0.10;
        double eta = 0.10;
        try {
            if (req.has_param("epsilon")) epsilon = std::stod(req.get_param_value("epsilon"));
            if (req.has_param("eta")) eta = std::stod(req.get_param_value("eta"));
        } catch (...) {
            epsilon = 0.10;
            eta = 0.10;
        }
        epsilon = std::max(0.0, std::min(1.0, epsilon));
        eta = std::max(0.0, std::min(1.0, eta));

        {
            std::lock_guard<std::mutex> lk(g_eval_mutex);
            if (g_eval_job.running) {
                res.status = 409;
                res.set_content("{\"error\":\"evaluation in progress\"}", "application/json");
                return;
            }
        }
        {
            std::lock_guard<std::mutex> lk(g_training_mutex);
            if (g_training_job.running) {
                res.status = 409;
                res.set_content("{\"error\":\"training already running\"}", "application/json");
                return;
            }
            g_training_job.running = true;
            g_training_job.requested_episodes = episodes;
            g_training_job.completed_episodes = 0;
            g_training_job.eval_interval = eval_interval;
            g_training_job.epsilon = epsilon;
            g_training_job.eta = eta;
            g_training_job.last_error.clear();
            g_training_job.last_result_json.clear();
            g_training_job.loss_history.clear();
            g_training_job.latest_q_loss = 0.0;
            g_training_job.latest_policy_loss = 0.0;
        }

        std::thread([episodes, eval_interval, epsilon, eta]() {
            try {
                auto result = run_training_report_json(
                    episodes, eval_interval, epsilon, eta,
                    [](int done, int total, const std::shared_ptr<SharedNFSPLearner>& learner) {
                        std::lock_guard<std::mutex> lk(g_training_mutex);
                        g_training_job.completed_episodes = std::min(done, total);
                        g_training_job.latest_q_loss = learner->latest_q_loss;
                        g_training_job.latest_policy_loss = learner->latest_policy_loss;
                        LossSample ls;
                        ls.episode = done;
                        ls.q_loss = learner->latest_q_loss;
                        ls.policy_loss = learner->latest_policy_loss;
                        g_training_job.loss_history.push_back(ls);
                        if (g_training_job.loss_history.size() > 500) {
                            g_training_job.loss_history.erase(
                                g_training_job.loss_history.begin(),
                                g_training_job.loss_history.begin() +
                                    static_cast<std::ptrdiff_t>(g_training_job.loss_history.size() - 500)
                            );
                        }
                    }
                );
                std::lock_guard<std::mutex> lk(g_training_mutex);
                g_training_job.last_result_json = result;
                g_training_job.completed_episodes = g_training_job.requested_episodes;
                g_training_job.running = false;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(g_training_mutex);
                g_training_job.last_error = e.what();
                g_training_job.running = false;
            } catch (...) {
                std::lock_guard<std::mutex> lk(g_training_mutex);
                g_training_job.last_error = "unknown training failure";
                g_training_job.running = false;
            }
        }).detach();

        res.set_content("{\"status\":\"started\"}", "application/json");
    });

    svr.Get("/api/train/status", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_training_mutex);
        std::ostringstream out;
        out << "{";
        out << "\"running\":" << (g_training_job.running ? "true" : "false") << ",";
        out << "\"requested_episodes\":" << g_training_job.requested_episodes << ",";
        out << "\"completed_episodes\":" << g_training_job.completed_episodes << ",";
        out << "\"latest_q_loss\":" << json_number(g_training_job.latest_q_loss) << ",";
        out << "\"latest_policy_loss\":" << json_number(g_training_job.latest_policy_loss) << ",";
        out << "\"loss_history\":[";
        for (size_t i = 0; i < g_training_job.loss_history.size(); ++i) {
            if (i) out << ",";
            const auto& ls = g_training_job.loss_history[i];
            out << "{\"episode\":" << ls.episode << ",\"q_loss\":" << json_number(ls.q_loss)
                << ",\"policy_loss\":" << json_number(ls.policy_loss) << "}";
        }
        out << "],";
        out << "\"error\":\"" << json_escape(g_training_job.last_error) << "\",";
        out << "\"has_result\":" << (!g_training_job.last_result_json.empty() ? "true" : "false") << ",";
        out << "\"result\":null";
        out << "}";
        res.set_content(out.str(), "application/json");
    });

    svr.Get("/api/train/result", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_training_mutex);
        if (g_training_job.last_result_json.empty()) {
            res.status = 404;
            res.set_content("{\"error\":\"no training result available\"}", "application/json");
            return;
        }
        res.set_content(
            std::string("{\"result\":") + g_training_job.last_result_json + "}",
            "application/json"
        );
    });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
