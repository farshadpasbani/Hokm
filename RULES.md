# Hokm Rules — As Implemented in This Codebase

Hokm (حکم) is a Persian trick-taking partnership game. Variants differ across
regions and tables. This document describes **exactly** the rules this engine
enforces, together with standard-rule deviations so the ML training and
evaluation are scientifically grounded.

All references below point at `hokm.py` at the state of this branch.

## 1. Players, teams, and seating

- **4 players**, fixed partnerships:
  - **Team 1**: seats 0 and 2 (`self.team1 = [players[0], players[2]]`)
  - **Team 2**: seats 1 and 3 (`self.team2 = [players[1], players[3]]`)
- Play direction: **clockwise by seat index** (`(i + 1) % 4`).
- In the web app (`app.py`), the human is **always seat 0 (South)**. The AI
  seats are 1=East, 2=North, 3=West. The human's teammate is therefore North.

## 2. Deck

- Standard **52 cards**, suits `{Hearts, Diamonds, Clubs, Spades}`,
  ranks `2..10, J, Q, K, A` with values `2..14`.
- Shuffled once per game (`Deck.shuffle`).

## 3. Dealing and Hakem selection

- On the **first** game, the Hakem is picked uniformly at random.
- On subsequent games, Hakem is determined by `rotate_hakem()` at the end of
  the previous game (see §7).
- Dealing happens in two phases:
  1. `start_game` deals **5 cards** to the Hakem face-down; no one else has
     cards yet.
  2. After the Hakem picks trump, `set_trump_suit` deals the remaining cards
     so that the Hakem ends up with **13** and each other player gets **13**
     (Hakem draws 8 more; others draw 13 fresh). Total cards dealt: 52.

> **Deviation note.** Real tables often deal in 5-4-4 segments with the Hakem
> bidding trump after the first 5. We collapse that into "Hakem sees 5,
> chooses trump, all hands complete to 13". This preserves the information
> model (Hakem picks trump with partial knowledge of own hand only).

## 4. Trump (Hokm) selection

- The Hakem chooses the trump suit.
- For AI / training, `Hokm.choose_trump_suit()` is a heuristic over the
  Hakem's 5 initial cards: pick the suit maximizing `count*10 + Σ weighted
  rank value` (high cards count double).
- For the web app, when the human is Hakem, the UI asks the human to pick;
  `/set_trump_suit` calls `Hokm.set_trump_suit(...)`.

## 5. Card play within a trick

- The **lead** to the first trick of a hand is the Hakem.
- The **lead** to each subsequent trick is the **winner** of the previous
  trick (`trick_starter_index = self.players.index(winner)`).
- The lead player may play **any** card from their hand; that card's suit is
  the **lead suit** for the trick (`self.lead_suit`).
- Each following player **must follow suit if possible**:
  - If the player has at least one card of `lead_suit`, they must play one
    of those cards (`legal_cards_for_player`).
  - If the player is void in `lead_suit`, they may play **any** card,
    including trump (Iranian Hokm allows but does not require trumping).

> **Strictness.** Revoking (not following suit when able) cannot happen in
> either play path, but the enforcement point differs: the web/API path
> (`apply_play`) rejects illegal cards at the engine level, while the
> training path (`play_round`) restricts the agent's *choice set* to legal
> cards inside `play_card`/`select_action` (with a random-legal fallback if
> an agent ever returns an out-of-set card).

## 6. Trick winner

Resolved by `determine_trick_winner()`:

1. If **any trump card** was played in this trick, the highest trump wins
   (higher `rank_value` among trump cards).
2. Otherwise, the **highest card of the lead suit** wins.
3. Off-suit, non-trump cards can never win.

## 7. Hand (round) termination and scoring

- A hand consists of **13 tricks** (each player plays all 13 cards).
- After each trick, the winning *team* increments `self.scores[team]`.
- **Hand ends as soon as** one team reaches **7 tricks** (`self.scores[t] >= 7`).
  The remaining cards are not played out.
- The team reaching 7 tricks first **wins the hand**.

> **Kot (کت) / Kapot (کپت).** Many tables count a 7-0 sweep differently
> (e.g. 2 or 3 "games" instead of 1, sometimes with the losing Hakem
> disqualified). **The engine does not model Kot** — its game-level win
> indicator is binary per hand. The Mini App *service layer*
> (`game_service.GameSession`) does: a 7-0 hand is flagged `kot` and worth
> **2 points** in the match score (see §9). Training and evaluation are
> unaffected.

## 8. Hakem rotation between hands

`rotate_hakem()` runs at the end of each hand:

- If the Hakem's team **did not win** the hand, the new Hakem is the
  `last_winning_team[0]` (the first listed winning-team seat — seat 0 for
  team 1, seat 1 for team 2).
- If the Hakem's team **did win** the hand, the Hakem seat **toggles**
  between the two seats of that team (e.g. seat 0 ↔ seat 2).

> **Deviation note.** At many tables the rule is simpler: "if your team wins,
> you stay Hakem; otherwise Hakem passes to the winning team's dealer-cut
> winner or to the team's declared seat". Our implementation is a mild
> simplification; because partnerships are fixed, it does not materially
> change the information model of the game.

## 9. Game termination (multi-hand match)

- The engine does **not** run matches. Each call to `Hokm.play_game()` plays
  **one hand**, declares a winning team, rotates Hakem, and returns.
- In the training loop and the evaluation loop, "one game" == "one hand".
- **Match play lives in `game_service.GameSession`** (Telegram Mini App):
  - `POST /api/new_game` starts a match at 0–0; `POST /api/next_hand` deals
    the next hand with the same four seats and the Hakem the engine rotated
    to at the end of the previous hand.
  - Winning a hand scores **1 point**, a Kot (7-0) scores **2**.
  - First team to `MATCH_TARGET` points (default **7**) wins the match.
  - Because the web app drives the engine through `apply_play` /
    `resolve_trick_if_complete` rather than `play_game()`, the service calls
    `update_last_winning_team()` + `rotate_hakem()` itself at hand end —
    before the next `start_game()`, which clears `tricks_won`.

## 10. Action & state representation used by the neural agent

- **Action space**: 52 discrete (card-index) actions. At each decision the
  agent may only choose among the **legal set** for that trick.
- **Observation (194-d, `EnhancedPlayer.get_state`)** — see
  `game_constants.STATE_LAYOUT` for the canonical index map:
  - 52-d one-hot: the agent's current hand
  - 52-d one-hot: all cards already played this hand (public memory)
  - 12-d: proven void flags per other player × suit (from failures to follow)
  - 52-d one-hot: cards on the table in the current in-progress trick
  - 4-d one-hot: lead suit; 4-d one-hot: trick position (1st–4th to play)
  - 5-d + 1-d: current trick winner (seat-relative) and winning card value
  - 2-d: Hakem-is-me / Hakem-is-partner flags
  - 2-d: team and opponent trick counts; 4-d one-hot: trump suit
  - 4-d: per-suit hand counts

> The observation is **not** a perfect-information state (other hands are
> never encoded): the agent plays with the same information a seated human
> player has — its own hand plus public history and inferences from it.

## 11. What we do **not** model

- No bidding beyond Hakem's trump pick.
- No double / redouble.
- No Kot / Kapot scoring **in the engine** (the Mini App service adds it —
  §7, §9).
- No match-level scoring **in the engine**; each `play_game()` is one hand.
- No chat, tells, or table talk.

---

## Rule invariants (tested in `tests/test_rules.py`)

1. Each player ends the hand with exactly 0 or some remaining cards such
   that total played + remaining = 52.
2. After `set_trump_suit`, every player has exactly 13 cards.
3. `lead_suit` matches the first card played each trick.
4. A player that has any card of `lead_suit` **cannot legally play off-suit**
   (`apply_play` rejects it).
5. Trump beats non-trump; higher trump beats lower trump; if no trump was
   played, highest lead-suit card wins.
6. `scores[1] + scores[2]` after a hand equals the number of tricks played
   (≤ 13).
7. The hand stops at 7 tricks for either team.
