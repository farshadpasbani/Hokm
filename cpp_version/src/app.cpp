#include "enhanced_player.h"
#include "hokm.h"
#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace hokm;

namespace {

enum class SeatMode { Human, AI };

class ConsolePlayer : public Player {
public:
    explicit ConsolePlayer(const std::string& name, SeatMode mode)
        : Player(name), mode_(mode) {}

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
};

struct SessionState {
    std::string id;
    int human_seat = 0;
    std::vector<std::shared_ptr<Player>> players;
    std::string ai_policy = "heuristic";
    std::shared_ptr<Hokm> game;
    std::vector<std::string> event_log;
};

std::unordered_map<std::string, SessionState> g_sessions;
std::mutex g_sessions_mutex;
std::mt19937 g_rng{std::random_device{}()};
std::shared_ptr<SharedNFSPLearner> g_trained_learner;
std::string g_trained_model_dir;
int g_trained_episodes = 0;

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
        (void)cp;
        reason = "console-heuristic";
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
};

double evaluate_trained_vs_heuristic(
    const std::shared_ptr<SharedNFSPLearner>& learner,
    int games
) {
    int trained_wins = 0;
    for (int g = 0; g < games; ++g) {
        // Alternate sides to reduce seat-order bias.
        const bool trained_on_team1 = (g % 2 == 0);
        std::vector<std::shared_ptr<Player>> players(4);
        for (int i = 0; i < 4; ++i) {
            const bool seat_team1 = (i == 0 || i == 2);
            const bool trained_seat = trained_on_team1 ? seat_team1 : !seat_team1;
            if (trained_seat) {
                auto p = std::make_shared<EnhancedPlayer>(
                    "Player " + std::to_string(i + 1), learner, 0.0, 1.0, "mixed"
                );
                p->learning_enabled = false;
                players[i] = p;
            } else {
                players[i] = std::make_shared<ConsolePlayer>(
                    "Player " + std::to_string(i + 1), SeatMode::AI
                );
            }
        }
        Hokm eval_game(players, "", true);
        eval_game.play_game(false);
        const bool team1_won = eval_game.scores[1] >= 7;
        const bool trained_won = trained_on_team1 ? team1_won : !team1_won;
        if (trained_won) trained_wins++;
    }
    return (games > 0) ? static_cast<double>(trained_wins) / games : 0.0;
}

std::string run_training_report_json(
    int episodes,
    int eval_interval,
    double epsilon,
    double eta
) {
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
    out << "\"eval_interval\":" << eval_interval << ",";
    out << "\"params\":{\"epsilon\":" << epsilon << ",\"eta\":" << eta << "},";
    out << "\"team1_wins\":" << t1_total << ",";
    out << "\"team2_wins\":" << t2_total << ",";
    out << "\"aborted_games\":" << game.aborted_games << ",";
    out << "\"elapsed_seconds\":" << secs << ",";
    out << "\"latest_q_loss\":" << learner->latest_q_loss << ",";
    out << "\"latest_policy_loss\":" << learner->latest_policy_loss << ",";
    out << "\"games_per_second\":" << (secs > 0 ? static_cast<double>(episodes) / secs : 0.0) << ",";
    out << "\"approach\":\"hybrid_basic_strategy_plus_nfsp_with_throttled_updates\",";
    out << "\"final_benchmark_win_rate_vs_heuristic\":"
        << (points.empty() ? 0.0 : points.back().benchmark_win_rate) << ",";
    out << "\"points\":[";
    for (size_t i = 0; i < points.size(); ++i) {
        if (i) out << ",";
        out << "{";
        out << "\"episode\":" << points[i].episode << ",";
        out << "\"team1_win_rate\":" << points[i].team1_win_rate << ",";
        out << "\"team2_win_rate\":" << points[i].team2_win_rate << ",";
        out << "\"avg_margin\":" << points[i].avg_margin << ",";
        out << "\"benchmark_win_rate\":" << points[i].benchmark_win_rate;
        out << "}";
    }
    out << "]";
    out << "}";

    // Publish this learner for real-scenario play sessions.
    g_trained_learner = learner;
    g_trained_episodes = episodes;
    g_trained_model_dir = "/Users/farshad/Documents/Documents - farshad’s MacBook Air/Projects/Coding/Hokm/Hokm/cpp_version/models/latest";
    g_trained_learner->save_models(g_trained_model_dir);
    return out.str();
}

} // namespace

int main() {
    httplib::Server svr;
    std::cout << "Starting Hokm C++ Web Server on http://localhost:8080..." << std::endl;

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        const std::string page = R"HTML(
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Hokm Dev Console</title>
  <style>
    body { margin: 0; font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b1020; color: #e5e7eb; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 16px; }
    .panel { background: #121a2f; border: 1px solid #273253; border-radius: 12px; padding: 14px; }
    h1 { margin: 0 0 8px 0; font-size: 24px; }
    h2 { margin: 0 0 10px 0; font-size: 16px; }
    .muted { color: #9ca3af; font-size: 13px; }
    button { border: 1px solid #3c4a76; background: #1a2443; color: #e5e7eb; border-radius: 8px; padding: 8px 10px; cursor: pointer; margin: 4px 4px 4px 0; }
    button:hover { background: #22315d; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .card-btn { display: inline-block; margin: 4px; }
    .row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    input, select { background: #0f1730; color: #e5e7eb; border: 1px solid #3c4a76; border-radius: 8px; padding: 6px 8px; }
    pre { background: #090f22; border: 1px solid #273253; border-radius: 8px; padding: 10px; max-height: 240px; overflow: auto; white-space: pre-wrap; }
    .kpi { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; margin: 10px 0; }
    .kpi > div { background: #0f1730; border: 1px solid #273253; border-radius: 8px; padding: 8px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Hokm Developer Console</h1>
    <div class="muted">Interactive human-vs-AI gameplay, strategy explainability, and configurable NFSP training diagnostics.</div>
    <div class="grid">
      <div class="panel">
        <h2>Session Controls</h2>
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
          <select id="aiPolicy">
            <option value="heuristic">Heuristic Basic Strategy</option>
            <option value="trained">Trained NFSP Model</option>
          </select>
        </div>
        <button onclick="newSession()">New Session</button>
        <button onclick="refreshState()">Refresh State</button>
        <button onclick="stepAI()">Step AI</button>
        <button onclick="autoPlay()">Auto Play to End</button>
        <div class="muted" id="sessionMeta">No active session.</div>

        <h2 style="margin-top:16px;">Training Lab</h2>
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
        <button onclick="train()">Run Advanced Training</button>
        <button onclick="train10k()">Run 10,000-Game Deep Run</button>
        <pre id="trainingOut">Not started.</pre>
        <canvas id="trainChart" width="300" height="160" style="width:100%;background:#090f22;border:1px solid #273253;border-radius:8px;"></canvas>
      </div>

      <div>
        <div class="panel">
          <h2>Live Game State</h2>
          <div class="kpi" id="kpis"></div>
          <div><strong>Current Trick</strong> <div id="trick"></div></div>
          <div style="margin-top:10px;"><strong>Your Legal Cards</strong></div>
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
<script>
let sessionId = "";
let state = null;

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

function updateView() {
  if (!state) return;
  document.getElementById("sessionMeta").textContent =
    `Session ${state.session_id} | next: ${state.next_player} | human turn: ${state.is_human_turn}`;

  document.getElementById("kpis").innerHTML = `
    <div><div class="muted">Trump</div><div>${state.trump_suit}</div></div>
    <div><div class="muted">Lead Suit</div><div>${state.lead_suit || "-"}</div></div>
    <div><div class="muted">Score</div><div>T1 ${state.scores.team1} - T2 ${state.scores.team2}</div></div>
    <div><div class="muted">Round</div><div>${state.round}</div></div>
  `;

  document.getElementById("trick").innerHTML = (state.current_trick || []).map(t =>
    `<div>${t.player}: <strong>${t.card}</strong></div>`
  ).join("") || "<span class='muted'>Empty trick.</span>";

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
  const data = await api(`/api/session/new?human_seat=${humanSeat}&ai_policy=${encodeURIComponent(aiPolicy)}`);
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
  document.getElementById("trainingOut").textContent = "Running...";
  const data = await api(`/api/train/advanced?episodes=${episodes}&eval_interval=${evalInterval}&epsilon=${epsilon}&eta=${eta}`);
  document.getElementById("trainingOut").textContent = JSON.stringify(data, null, 2);
  renderTrainingChart(data.points || []);
}

async function train10k() {
  document.getElementById("episodes").value = 10000;
  document.getElementById("evalInterval").value = 1000;
  await train();
}

function renderTrainingChart(points) {
  const canvas = document.getElementById("trainChart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#090f22";
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
        out << "\"model_dir\":\"" << json_escape(g_trained_model_dir) << "\"";
        out << "}";
        res.set_content(out.str(), "application/json");
    });

    svr.Get("/api/session/new", [](const httplib::Request& req, httplib::Response& res) {
        int human_seat = 0;
        std::string ai_policy = "heuristic";
        if (req.has_param("human_seat")) {
            human_seat = clamp_int(parse_int_or(req.get_param_value("human_seat"), 0), 0, 3);
        }
        if (req.has_param("ai_policy")) {
            ai_policy = req.get_param_value("ai_policy");
        }
        if (ai_policy != "heuristic" && ai_policy != "trained") ai_policy = "heuristic";
        if (ai_policy == "trained" && !g_trained_learner) {
            ai_policy = "heuristic";
        }

        SessionState s;
        s.id = make_session_id();
        s.human_seat = human_seat;
        s.ai_policy = ai_policy;
        s.players.resize(4);
        for (int i = 0; i < 4; ++i) {
            if (i == human_seat) {
                s.players[i] = std::make_shared<ConsolePlayer>("Player " + std::to_string(i + 1), SeatMode::Human);
            } else if (ai_policy == "trained" && g_trained_learner) {
                auto p = std::make_shared<EnhancedPlayer>(
                    "Player " + std::to_string(i + 1), g_trained_learner, 0.0, 1.0, "mixed"
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

    svr.listen("0.0.0.0", 8080);
    return 0;
}
