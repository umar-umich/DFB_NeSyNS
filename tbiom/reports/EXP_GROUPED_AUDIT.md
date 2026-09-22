# Grouped-fusion algebraic and mechanism checks (Stage 7)

Every operator exercised on constructed opinions where the answer is known independently of any dataset. Run before dataset scoring, so an implementation error surfaces here.

**28 of 28 checks pass.**

| check | result | detail |
|---|---|---|
| constructed opinions are admissible | pass | 0 violations |
| b1 + b0 + u = 1 | pass | max dev 2.22e-16 |
| p recovered from the opinion | pass | max dev 0.00e+00 |
| evidence round trip e -> opinion -> e | pass | max dev 1.11e-16 |
| alpha_k = e_k + 1 and u = 2/sum(alpha) | pass | max dev 1.11e-16 |
| evidence scaling is e' = r e, not belief-mass discounting | pass | belief-mass discounting would give b' = r b, u' = 1 - r(1-u) |
| Dempster: masses non-negative | pass | min 6.25e-04 |
| Dempster: masses sum to 1 | pass | max dev 7.77e-16 |
| Dempster: two-source symmetry | pass | max dev 2.22e-16 |
| Yager: masses non-negative | pass | min 2.57e-03 |
| Yager: masses sum to 1 | pass | max dev 2.22e-16 |
| Yager: two-source symmetry | pass | max dev 2.22e-16 |
| PCR6: masses non-negative | pass | min 4.23e-04 |
| PCR6: masses sum to 1 | pass | max dev 4.44e-16 |
| PCR6: two-source symmetry | pass | max dev 2.22e-16 |
| cautious: masses sum to 1 | pass | max dev 1.55e-15 |
| cautious: idempotent on admissible opinions | pass | max dev 6.66e-16 |
| cautious: exchange invariant | pass |  |
| cautious: duplicating a view inside the group changes nothing | pass | the property Dempster does not have |
| Dempster: duplicating a view DOES change the result (the double count being avoided) | pass |  |
| evidence average: adds the Dirichlet prior once | pass | an average of probabilities would not satisfy this |
| vacuous is the Dempster identity | pass | max dev 0.00e+00 |
| grouped-R with a vacuous temporal view reduces to the cautious spatial pair | pass |  |
| grouped-R with an absent temporal view reduces to the same | pass | absence handled as pass-through |
| grouped-R is not three-source Dempster | pass |  |
| grouped-R is not the global cautious rule | pass |  |
| grouped-R is APP/OM exchange invariant | pass |  |
| PCR6 keeps near-zero vacuity under opposite strong opinions | pass | u_PCR6 = 4.00e-04 against u_Dempster = 6.78e-03 — the brief's warning, confirmed |

## Limit cases, p_fused by rule

| case | Dempster | Yager | PCR6 | cautious | u Dempster | u PCR6 |
|---|---:|---:|---:|---:|---:|---:|
| matching strong | 0.9989 | 0.9869 | 0.9986 | 0.9800 | 6.15e-04 | 6.00e-04 |
| opposite strong | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 6.78e-03 | 4.00e-04 |
| one strong one weak | 0.9816 | 0.9575 | 0.9800 | 0.9800 | 1.68e-02 | 1.60e-02 |
| very small vacuity | 0.9808 | 0.8750 | 0.9568 | 0.8571 | 1.28e-12 | 1.00e-12 |

The `opposite strong` row is the brief's warning made concrete: PCR6 redistributes the conflicting mass to the classes and leaves the vacuity at the product of two near-zero vacuities, so it returns a near-0.5 probability with almost no vacuity — confidently undecided. Dempster returns the same probability with the conflict visible. Any arm with a PCR6 outer step is therefore at risk on the uncertainty endpoint by construction, which is why the pre-registration disqualifies an arm on vacuity error-ranking regardless of its detection number.

PCR6 is not associative, so source-family metadata is preserved throughout and no regrouping invariance is claimed for the PCR6-outer arms.

