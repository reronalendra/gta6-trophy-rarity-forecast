# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/reronalendra/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/reronalendra/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01** — forecast each of Grand Theft Auto VI's 55 PS5 trophies' Exophase earn rate at the case's target time, as five quantiles (5/25/50/75/95%) per trophy.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case from `/input` (or `$MAKEFUN_INPUT_DIR`) and writes the matching `*.out` forecast to `/output` (or `$MAKEFUN_OUTPUT_DIR`). Python 3.13, standard library only, no network, deterministic.

## Method (original work)

A trophy's rate is **earners / players_tracked**, and the two move for different reasons, so they are forecast separately and divided:

- **Players P(t):** power law on all reads, power law on the last 24h, and a logistic in log-time (k=1), combined by a blend that trusts the power law when observed growth is moderate and the saturating logistic when it is steep — launch-week tracking growth is explosive early and saturates later. Reads before release are dropped from the fits.
- **Earners E_i(t):** log-log fit of the earner counts reconstructed from each read (E = rate × P / 100), slope shrunk toward the cross-trophy median slope, anchored at the last reliable point. Rates below 0.03% are dropped from the fit: at the 2-decimal rounding floor the reconstructed earner count is dominated by rounding (it grows like P, not like the true earners).
- **Median rate per trophy:** either the logistic share s(t) = S / (1 + (t50/t)^k) fitted to the trophy's observed rates (when its share has been rising strongly, or rose a lot and then flattened — the saturation signature — or the counts median would imply an implausible share rise while the share is flat), or E_i(target)/P(target) otherwise (dilution dominates).
- **Spread:** honest and wide where the fits disagree or the history is thin, narrow where the counts are stable and well observed.
- **Flat history** (every read identical, as in the practice cases): Exophase refreshes its page in jumps; the documented launch-night refresh took players 638 → 3868 with earners unchanged, so one such refresh is assumed before the target, earners conserved, the 100% trophy kept at 100%, with a narrow spread.

## Local validation

- **Practice cases** (answers reconstructed from the documented launch-night refresh, earners conserved, 638 → 3868 players; reconstruction validated: implied earners sum to the recorded 943): mean pinball loss **0.011532** (house persistence baseline: 1.688).
- **36 synthetic final cases** (dynamics calibrated from the real launch rates and the live Exophase page of 2026-11-20, spanning players-tracked at the live page 100k/200k/400k, share-curve shapes k=1.0/1.5/2.0, saturation levels 1.5M/3M players, refresh cadences 4h/8h): mean loss **0.140**, median 0.127, worst 0.218 — beats the house persistence baseline (0.271) and the house curve baseline (0.310) in **36/36** scenarios.
- Deterministic (two runs byte-identical), ~0.7 s per final case, 1.15 s stress test on 120 reads (limit 8 s), stdlib only, no network. Edge cases handled: all-None reads, single-read history, all-zero trophies, 100% trophy kept at 100%.

---

<sub>Entered on [make.fun](https://make.fun/?ref=2s61jrxf&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
