# What the evaluation scope costs — CDFv3 and KoDF

These are the only two test sets whose video census the scope touches. The other ten
contain none of the excluded families and are byte-identical under every scope.

The system is unchanged across the rows: `App + TD + LFS + STT` under PCR6+-R with the
weights fitted once on the development pool, which no scope touches. A difference below
is a difference in which videos are scored, and nothing else.

## CDFv3

| scope | videos (real / fake) | AUROC | FPR@TPR90 | App | TD | LFS | STT |
|---|---|---:|---:|---:|---:|---:|---:|
| v3.1 (published) | 178 / 4940 | **0.9612** | 0.1124 | 0.8935 | 0.9263 | 0.8801 | 0.8864 |
| + lip-sync / audio-driven | 178 / 5240 | **0.9614** | 0.1124 | 0.8893 | 0.9288 | 0.8790 | 0.8884 |
| + adversarial | 178 / 5240 | **0.9614** | 0.1124 | 0.8893 | 0.9288 | 0.8790 | 0.8884 |

`nolip` admits: `synthesis-TalkingFace-IP_LAP`

## KoDF

| scope | videos (real / fake) | AUROC | FPR@TPR90 | App | TD | LFS | STT |
|---|---|---:|---:|---:|---:|---:|---:|
| v3.1 (published) | 403 / 992 | **0.9436** | 0.0397 | 0.8563 | 0.9656 | 0.8548 | 0.9268 |
| + lip-sync / audio-driven | 403 / 1106 | **0.9494** | 0.0323 | 0.8692 | 0.9653 | 0.8647 | 0.9335 |
| + adversarial | 806 / 1433 | **0.9112** | 0.3834 | 0.8117 | 0.9298 | 0.7999 | 0.9093 |

`nolip` admits: `fake-audio-driven`
`nolipadv` admits: `adv-fake`, `adv-real`

## Where the false alarms come from

The threshold is the one that catches 90% of the fakes on the census WITHOUT the
adversarial subsets — the system as it would be deployed having never seen them — and
each genuine subset is then asked what fraction of it is flagged at that threshold.
This separates two things an aggregate FPR cannot: new fakes being missed, and new
genuine videos being accused.

### CDFv3

| subset | label | videos | flagged at that threshold |
|---|---|---:|---:|
| `YouTube-real` | genuine | 70 | 0.1286 |
| `Celeb-real` | genuine | 108 | 0.1019 |
| `synthesis-FaceSwap-SimSwap` | fake | 200 | 0.5700 |
| `synthesis-TalkingFace-SadTalker` | fake | 300 | 0.6333 |
| `synthesis-FaceReenact-DaGAN` | fake | 200 | 0.7500 |
| `synthesis-FaceReenact-LivePortrait` | fake | 200 | 0.8350 |
| `synthesis-TalkingFace-AniTalker` | fake | 300 | 0.8533 |
| `synthesis-TalkingFace-EchoMimic` | fake | 300 | 0.8567 |
| `synthesis-FaceReenact-FSRT` | fake | 200 | 0.8650 |
| `synthesis-FaceReenact-MCNET` | fake | 200 | 0.9100 |
| `synthesis-FaceReenact-TPSMM` | fake | 200 | 0.9100 |
| `synthesis-TalkingFace-IP_LAP` | fake | 300 | 0.9100 |
| `synthesis-FaceSwap-GHOST` | fake | 200 | 0.9400 |
| `synthesis-FaceSwap-MobileFaceSwap` | fake | 200 | 0.9450 |
| `synthesis-TalkingFace-FLOAT` | fake | 300 | 0.9467 |
| `synthesis-FaceSwap-Celeb-DF-v2` | fake | 340 | 0.9529 |
| `synthesis-FaceReenact-HyperReenact` | fake | 200 | 0.9700 |
| `synthesis-FaceReenact-LIA` | fake | 200 | 0.9700 |
| `synthesis-FaceSwap-HifiFace` | fake | 200 | 0.9950 |
| `synthesis-FaceSwap-BlendFace` | fake | 200 | 1.0000 |
| `synthesis-FaceSwap-InSwapper` | fake | 200 | 1.0000 |
| `synthesis-FaceSwap-UniFace` | fake | 200 | 1.0000 |
| `synthesis-TalkingFace-EDTalk` | fake | 300 | 1.0000 |
| `synthesis-TalkingFace-R3DPortrait` | fake | 300 | 1.0000 |

### KoDF

| subset | label | videos | flagged at that threshold |
|---|---|---:|---:|
| `adv-real` | genuine | 403 | 0.0546 |
| `real` | genuine | 403 | 0.0323 |
| `adv-fake` | fake | 327 | 0.4128 |
| `fake-fsgan` | fake | 205 | 0.6195 |
| `fake-fo` | fake | 352 | 0.9062 |
| `fake-audio-driven` | fake | 114 | 1.0000 |
| `fake-dffs` | fake | 223 | 1.0000 |
| `fake-dfl` | fake | 212 | 1.0000 |

At this fixed threshold the adversarial genuine videos are flagged at 0.0546 against 0.0323 for the ordinary ones — close. What moves the aggregate is the fake side: `adv-fake` is caught at only 0.4128, so holding TPR at 90% over the wider census forces a much lower threshold, and that is what floods the genuine videos.

