# Step 5, second basis — DF40-Dev against P0-DS

VALmix returned the verdict at 100% join coverage; this confirms it on the generator zoo, the same probe the earlier CLIP-port gate used.

**34 of 36 methods.** `simswap` and `uniface` (DF40's bare 'unknown' arms) have no DiCoME config under those names — only `simswap_cdf/_ff` and `uniface_cdf/_ff` exist — so they are absent and recorded rather than silently dropped.

| expert | videos | methods | rescue | harm | margin | ceiling | recovered | AUROC | enters? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fsvfm_preserve` | 18575 | 34 | 0.282 | 0.223 | +0.060 | 0.833 | 3.8% | 0.8314 → 0.8407 | no |
| `fsvfm_ordinary` | 18575 | 34 | 0.256 | 0.231 | +0.025 | 0.827 | 1.4% | 0.8314 → 0.8386 | no |

## Verdict

**Confirmed — no expert enters on DF40-Dev either.** That makes FOUR bases in agreement: DF40-Dev and VALmix against the CLIP port, and VALmix and now DF40-Dev against the stronger P0-DS anchor. The membership negative is not an artifact of one probe or one anchor.
