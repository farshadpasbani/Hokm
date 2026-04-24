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

> **Strictness.** Revoking (not following suit when able) is **illegal** at
> the engine level — `apply_play` rejects it. In training, the agent can only
> ever *choose* from legal actions (the reward function still includes a
> penalty for deliberately picking a non-following card, but the engine
> enforces legality regardless).

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
> disqualified). **We do not model Kot.** The game-level win indicator is
> binary per hand. A `kot` flag can be added later; the engine already tracks
> `tricks_won` per player so the information is available.

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

- The engine does **not** currently run matches to some target score of
  hands (e.g. best-of-7). Each call to `Hokm.play_game()` plays **one hand**,
  declares a winning team, rotates Hakem, and returns.
- In the training loop and the evaluation loop, "one game" == "one hand".

## 10. Action & state representation used by the neural agent

- **Action space**: 52 discrete (card-index) actions. At each decision the
  engine restricts to the **legal set** for that trick.
- **Observation (114-d, `EnhancedPlayer.get_state`)**:
  - 52-d one-hot: the agent's current hand
  - 4-d: played-cards-by-suit counter (from this agent's perspective)
  - 52-d one-hot: the **last** card played in the current trick
    *(known limitation — see ARCHITECTURE.md)*
  - 2-d: team tricks so far, opponent tricks so far
  - 4-d one-hot: trump suit

> The observation is **not** a sufficient perfect-information state (we don't
> model other hands) and is also **not** a complete Markov view of the trick
> (only the most recent card is encoded). Both are intentional: the agent
> plays with the same information a seated human player has, and the trick
> summary is deliberately coarse. A richer Markov observation is listed as
> an improvement in `ARCHITECTURE.md`.

## 11. What we do **not** model

- No bidding beyond Hakem's trump pick.
- No double / redouble.
- No Kot / Kapot scoring as noted in §7.
- No match-level scoring; each `play_game()` is one hand.
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
