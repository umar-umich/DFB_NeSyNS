# FF++ (+) DF40 combined training manifest

`/data/umar/Datasets/preprocessed/dataset_json/FFPP_DF40.json`  ·  label keys `FFPP_DF40_Real` = 0, `FFPP_DF40_Fake` = 1

| half | videos | frames |
|---|---:|---:|
| real | 720 | 23039 |
| fake | 23668 | 753570 |

Fakes are built from DF40's PER-METHOD `*_ff.json` files, not `DF40_all.json`. The aggregate leaks FF++ test identities into its train split (4,031 e4e images and 258 simswap/inswap videos) and its e4e paths do not resolve. Every video here is additionally filtered against the FF++ val and test identity lists, dropping on EITHER id of a `<target>_<source>` pair.

`dropped: empty` are entries DF40 ships with an empty frame list (generation failed for that id); they are NOT path-resolution failures and are counted apart so the two are never confused.

| source | videos | frames | dropped: identity | dropped: empty | dropped: unresolved |
|---|---:|---:|---:|---:|---:|
| `FF++/FF-real` | 720 | 23039 | 0 | 0 | 0 |
| `FF++/FF-DF` | 720 | 23039 | 0 | 0 | 0 |
| `FF++/FF-F2F` | 720 | 23040 | 0 | 0 | 0 |
| `FF++/FF-FS` | 720 | 23040 | 0 | 0 | 0 |
| `FF++/FF-NT` | 720 | 23040 | 0 | 0 | 0 |
| `DF40/DiT_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/MRAA_ff` | 718 | 22811 | 0 | 0 | 0 |
| `DF40/SiT_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/StyleGAN2_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/StyleGAN3_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/StyleGANXL_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/VQGAN_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/blendface_ff` | 711 | 22513 | 0 | 0 | 0 |
| `DF40/danet_ff` | 719 | 22646 | 0 | 0 | 0 |
| `DF40/ddim_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/e4e_ff` | 719 | 22993 | 0 | 1 | 0 |
| `DF40/e4s_ff` | 718 | 22779 | 0 | 0 | 0 |
| `DF40/facedancer_ff` | 716 | 22747 | 0 | 0 | 0 |
| `DF40/faceswap_ff` | 720 | 22852 | 0 | 0 | 0 |
| `DF40/facevid2vid_ff` | 716 | 22815 | 0 | 0 | 0 |
| `DF40/fomm_ff` | 719 | 22811 | 0 | 0 | 0 |
| `DF40/fsgan_ff` | 686 | 21815 | 0 | 0 | 0 |
| `DF40/hyperreenact_ff` | 587 | 18783 | 0 | 0 | 0 |
| `DF40/inswap_ff` | 664 | 21017 | 219 | 0 | 0 |
| `DF40/lia_ff` | 719 | 22811 | 0 | 0 | 0 |
| `DF40/mcnet_ff` | 719 | 22613 | 0 | 0 | 0 |
| `DF40/mobileswap_ff` | 718 | 22824 | 0 | 0 | 0 |
| `DF40/one_shot_free_ff` | 718 | 22959 | 0 | 0 | 0 |
| `DF40/pirender_ff` | 717 | 22914 | 0 | 0 | 0 |
| `DF40/pixart_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/rddm_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/sadtalker_ff` | 703 | 22413 | 12 | 0 | 0 |
| `DF40/sd2.1_ff` | 564 | 18007 | 0 | 155 | 0 |
| `DF40/simswap_ff` | 713 | 22609 | 275 | 0 | 0 |
| `DF40/tpsm_ff` | 719 | 22964 | 0 | 0 | 0 |
| `DF40/uniface_ff` | 318 | 10098 | 54 | 0 | 0 |
| `DF40/wav2lip_ff` | 711 | 22554 | 4 | 0 | 0 |
