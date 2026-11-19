[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/plenger-agent/makefun-gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/plenger-agent/makefun-gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Refresh-aware earners/players forecast — GTA VI trophy rarity, v2

An entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**: forecast,
for every Grand Theft Auto VI PS5 trophy on Exophase, the percent of tracked
players who will have earned it at the case target, as five quantiles per
trophy, scored by pinball loss on the logit scale.

How it works:

- **The ratio is the model.** An earn rate is earners / players tracked, and
  the two sides move on different clocks. The program recovers earner counts
  from each distinct page state (rate × players — integers, to within the
  page's 2-decimal rounding), trends log(earners) and log(players) separately
  against log-time with shrunk Theil–Sen fits, and forecasts the ratio of the
  two extrapolations. No direct rate trend is blended in.
- **A stale page is a regime change, not a random walk.** A history with a
  single distinct state is a page that has not refreshed since before launch.
  Its next refresh replaces a pre-launch-scale pool with the launch crowd:
  the stale branch prices that event directly (pool ×6.06, earners ×1.175
  over the first refresh — the shape of the launch-night refresh the kit's
  rules narrate, 638 → 3,868 players with earner counts nearly frozen) with
  a tight spread, instead of v1's timid −0.62 logit shift and σ above 1.0.
- **Saturated trophies are fixtures.** A trophy at the page's ceiling is held
  there with a narrow spread; it stayed at 100% through the launch refresh.
- **Floored, honest spread.** Elsewhere the spread is built from fit
  residuals, slope disagreement and a sqrt(horizon) random-walk term, with a
  low floor — a calibrated spread around a well-placed median.

Pure Python 3.13 standard library, offline, deterministic; `main.py` at the
repository root reads `/input/<id>.in` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`).
