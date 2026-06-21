# Feature-vector layout (the 695-dim per-window vector)

> Auto-generated from the code (`features.build_features` + `dataset.windows.aggregate_feature_names`). This is the exact input the GBT baseline
> trains on, and the vector the hybrid ST-MoE injects via `eng_proj` (`models/network.py`). Regenerate if the feature pipeline changes.

## Structure

Each window of 139 engineered features is summarised by 5 aggregates `mean, std, min, max, last` → **139 × 5 = 695** values.

The vector is **stat-major**: five consecutive blocks of 139, one per aggregate. The same 139 features appear in the same order in each block.

```
positions   0 .. 138   ->  __mean
positions 139 .. 277   ->  __std
positions 278 .. 416   ->  __min
positions 417 .. 555   ->  __max
positions 556 .. 694   ->  __last
```

**Lookup:** `position = stat_index * 139 + feature_index`  (`mean=0, std=1, min=2, max=3, last=4`)

## The 139 base features (feature_index) and where their 5 aggregates land

| idx | feature | family | mean | std | min | max | last |
|----:|---------|--------|----:|----:|----:|----:|----:|
| 0 | `dP_CKP` | differentials | 0 | 139 | 278 | 417 | 556 |
| 1 | `dP_CKGL` |  | 1 | 140 | 279 | 418 | 557 |
| 2 | `dT_CKP` |  | 2 | 141 | 280 | 419 | 558 |
| 3 | `dP_grad` |  | 3 | 142 | 281 | 420 | 559 |
| 4 | `dT_grad` |  | 4 | 143 | 282 | 421 | 560 |
| 5 | `dP_tree` |  | 5 | 144 | 283 | 422 | 561 |
| 6 | `dP_grad_roc` |  | 6 | 145 | 284 | 423 | 562 |
| 7 | `dP_CKP_roc` |  | 7 | 146 | 285 | 424 | 563 |
| 8 | `Cv_CKGL` | choke_coeff | 8 | 147 | 286 | 425 | 564 |
| 9 | `cond_CKP` |  | 9 | 148 | 287 | 426 | 565 |
| 10 | `Cv_BS` |  | 10 | 149 | 288 | 427 | 566 |
| 11 | `hydrate_margin_prod` | hydrate_margin | 11 | 150 | 289 | 428 | 567 |
| 12 | `hydrate_ratio_prod` |  | 12 | 151 | 290 | 429 | 568 |
| 13 | `hydrate_margin_svc` |  | 13 | 152 | 291 | 430 | 569 |
| 14 | `P-PDG__dom_freq` | spectral | 14 | 153 | 292 | 431 | 570 |
| 15 | `P-PDG__spec_entropy` |  | 15 | 154 | 293 | 432 | 571 |
| 16 | `P-PDG__band_0.0_0.01` |  | 16 | 155 | 294 | 433 | 572 |
| 17 | `P-PDG__band_0.01_0.05` |  | 17 | 156 | 295 | 434 | 573 |
| 18 | `P-PDG__band_0.05_0.2` |  | 18 | 157 | 296 | 435 | 574 |
| 19 | `P-TPT__dom_freq` |  | 19 | 158 | 297 | 436 | 575 |
| 20 | `P-TPT__spec_entropy` |  | 20 | 159 | 298 | 437 | 576 |
| 21 | `P-TPT__band_0.0_0.01` |  | 21 | 160 | 299 | 438 | 577 |
| 22 | `P-TPT__band_0.01_0.05` |  | 22 | 161 | 300 | 439 | 578 |
| 23 | `P-TPT__band_0.05_0.2` |  | 23 | 162 | 301 | 440 | 579 |
| 24 | `PT-P__dom_freq` |  | 24 | 163 | 302 | 441 | 580 |
| 25 | `PT-P__spec_entropy` |  | 25 | 164 | 303 | 442 | 581 |
| 26 | `PT-P__band_0.0_0.01` |  | 26 | 165 | 304 | 443 | 582 |
| 27 | `PT-P__band_0.01_0.05` |  | 27 | 166 | 305 | 444 | 583 |
| 28 | `PT-P__band_0.05_0.2` |  | 28 | 167 | 306 | 445 | 584 |
| 29 | `QGL__dom_freq` |  | 29 | 168 | 307 | 446 | 585 |
| 30 | `QGL__spec_entropy` |  | 30 | 169 | 308 | 447 | 586 |
| 31 | `QGL__band_0.0_0.01` |  | 31 | 170 | 309 | 448 | 587 |
| 32 | `QGL__band_0.01_0.05` |  | 32 | 171 | 310 | 449 | 588 |
| 33 | `QGL__band_0.05_0.2` |  | 33 | 172 | 311 | 450 | 589 |
| 34 | `P-PDG__osc_amp` | oscillation | 34 | 173 | 312 | 451 | 590 |
| 35 | `P-PDG__osc_std` |  | 35 | 174 | 313 | 452 | 591 |
| 36 | `P-PDG__osc_period` |  | 36 | 175 | 314 | 453 | 592 |
| 37 | `P-TPT__osc_amp` |  | 37 | 176 | 315 | 454 | 593 |
| 38 | `P-TPT__osc_std` |  | 38 | 177 | 316 | 455 | 594 |
| 39 | `P-TPT__osc_period` |  | 39 | 178 | 317 | 456 | 595 |
| 40 | `PT-P__osc_amp` |  | 40 | 179 | 318 | 457 | 596 |
| 41 | `PT-P__osc_std` |  | 41 | 180 | 319 | 458 | 597 |
| 42 | `PT-P__osc_period` |  | 42 | 181 | 320 | 459 | 598 |
| 43 | `QGL__osc_amp` |  | 43 | 182 | 321 | 460 | 599 |
| 44 | `QGL__osc_std` |  | 44 | 183 | 322 | 461 | 600 |
| 45 | `QGL__osc_period` |  | 45 | 184 | 323 | 462 | 601 |
| 46 | `P-MON-CKP__osc_amp` |  | 46 | 185 | 324 | 463 | 602 |
| 47 | `P-MON-CKP__osc_std` |  | 47 | 186 | 325 | 464 | 603 |
| 48 | `P-MON-CKP__osc_period` |  | 48 | 187 | 326 | 465 | 604 |
| 49 | `ESTADO-M1__delta` | valve_transitions | 49 | 188 | 327 | 466 | 605 |
| 50 | `ESTADO-M1__changed` |  | 50 | 189 | 328 | 467 | 606 |
| 51 | `ESTADO-M1__time_since_change` |  | 51 | 190 | 329 | 468 | 607 |
| 52 | `ESTADO-W1__delta` |  | 52 | 191 | 330 | 469 | 608 |
| 53 | `ESTADO-W1__changed` |  | 53 | 192 | 331 | 470 | 609 |
| 54 | `ESTADO-W1__time_since_change` |  | 54 | 193 | 332 | 471 | 610 |
| 55 | `ESTADO-SDV-P__delta` |  | 55 | 194 | 333 | 472 | 611 |
| 56 | `ESTADO-SDV-P__changed` |  | 56 | 195 | 334 | 473 | 612 |
| 57 | `ESTADO-SDV-P__time_since_change` |  | 57 | 196 | 335 | 474 | 613 |
| 58 | `ESTADO-DHSV__delta` |  | 58 | 197 | 336 | 475 | 614 |
| 59 | `ESTADO-DHSV__changed` |  | 59 | 198 | 337 | 476 | 615 |
| 60 | `ESTADO-DHSV__time_since_change` |  | 60 | 199 | 338 | 477 | 616 |
| 61 | `ESTADO-M2__delta` |  | 61 | 200 | 339 | 478 | 617 |
| 62 | `ESTADO-M2__changed` |  | 62 | 201 | 340 | 479 | 618 |
| 63 | `ESTADO-M2__time_since_change` |  | 63 | 202 | 341 | 480 | 619 |
| 64 | `ESTADO-W2__delta` |  | 64 | 203 | 342 | 481 | 620 |
| 65 | `ESTADO-W2__changed` |  | 65 | 204 | 343 | 482 | 621 |
| 66 | `ESTADO-W2__time_since_change` |  | 66 | 205 | 344 | 483 | 622 |
| 67 | `ESTADO-SDV-GL__delta` |  | 67 | 206 | 345 | 484 | 623 |
| 68 | `ESTADO-SDV-GL__changed` |  | 68 | 207 | 346 | 485 | 624 |
| 69 | `ESTADO-SDV-GL__time_since_change` |  | 69 | 208 | 347 | 486 | 625 |
| 70 | `ESTADO-XO__delta` |  | 70 | 209 | 348 | 487 | 626 |
| 71 | `ESTADO-XO__changed` |  | 71 | 210 | 349 | 488 | 627 |
| 72 | `ESTADO-XO__time_since_change` |  | 72 | 211 | 350 | 489 | 628 |
| 73 | `ESTADO-PXO__delta` |  | 73 | 212 | 351 | 490 | 629 |
| 74 | `ESTADO-PXO__changed` |  | 74 | 213 | 352 | 491 | 630 |
| 75 | `ESTADO-PXO__time_since_change` |  | 75 | 214 | 353 | 492 | 631 |
| 76 | `ABER-CKP__norm` | normalise | 76 | 215 | 354 | 493 | 632 |
| 77 | `P-MON-CKP__norm` |  | 77 | 216 | 355 | 494 | 633 |
| 78 | `P-JUS-CKP__norm` |  | 78 | 217 | 356 | 495 | 634 |
| 79 | `T-MON-CKP__norm` |  | 79 | 218 | 357 | 496 | 635 |
| 80 | `T-JUS-CKP__norm` |  | 80 | 219 | 358 | 497 | 636 |
| 81 | `P-MON-SDV-P__norm` |  | 81 | 220 | 359 | 498 | 637 |
| 82 | `PT-P__norm` |  | 82 | 221 | 360 | 499 | 638 |
| 83 | `P-TPT__norm` |  | 83 | 222 | 361 | 500 | 639 |
| 84 | `T-TPT__norm` |  | 84 | 223 | 362 | 501 | 640 |
| 85 | `P-PDG__norm` |  | 85 | 224 | 363 | 502 | 641 |
| 86 | `T-PDG__norm` |  | 86 | 225 | 364 | 503 | 642 |
| 87 | `ABER-CKGL__norm` |  | 87 | 226 | 365 | 504 | 643 |
| 88 | `QGL__norm` |  | 88 | 227 | 366 | 505 | 644 |
| 89 | `P-ANULAR__norm` |  | 89 | 228 | 367 | 506 | 645 |
| 90 | `P-MON-CKGL__norm` |  | 90 | 229 | 368 | 507 | 646 |
| 91 | `P-JUS-CKGL__norm` |  | 91 | 230 | 369 | 508 | 647 |
| 92 | `QBS__norm` |  | 92 | 231 | 370 | 509 | 648 |
| 93 | `P-JUS-BS__norm` |  | 93 | 232 | 371 | 510 | 649 |
| 94 | `ABER-CKP__is_missing` | quality · is_missing | 94 | 233 | 372 | 511 | 650 |
| 95 | `P-MON-CKP__is_missing` |  | 95 | 234 | 373 | 512 | 651 |
| 96 | `P-JUS-CKP__is_missing` |  | 96 | 235 | 374 | 513 | 652 |
| 97 | `T-MON-CKP__is_missing` |  | 97 | 236 | 375 | 514 | 653 |
| 98 | `T-JUS-CKP__is_missing` |  | 98 | 237 | 376 | 515 | 654 |
| 99 | `P-MON-SDV-P__is_missing` |  | 99 | 238 | 377 | 516 | 655 |
| 100 | `PT-P__is_missing` |  | 100 | 239 | 378 | 517 | 656 |
| 101 | `P-TPT__is_missing` |  | 101 | 240 | 379 | 518 | 657 |
| 102 | `T-TPT__is_missing` |  | 102 | 241 | 380 | 519 | 658 |
| 103 | `P-PDG__is_missing` |  | 103 | 242 | 381 | 520 | 659 |
| 104 | `T-PDG__is_missing` |  | 104 | 243 | 382 | 521 | 660 |
| 105 | `ABER-CKGL__is_missing` |  | 105 | 244 | 383 | 522 | 661 |
| 106 | `QGL__is_missing` |  | 106 | 245 | 384 | 523 | 662 |
| 107 | `P-ANULAR__is_missing` |  | 107 | 246 | 385 | 524 | 663 |
| 108 | `P-MON-CKGL__is_missing` |  | 108 | 247 | 386 | 525 | 664 |
| 109 | `P-JUS-CKGL__is_missing` |  | 109 | 248 | 387 | 526 | 665 |
| 110 | `QBS__is_missing` |  | 110 | 249 | 388 | 527 | 666 |
| 111 | `P-JUS-BS__is_missing` |  | 111 | 250 | 389 | 528 | 667 |
| 112 | `ESTADO-M1__is_missing` |  | 112 | 251 | 390 | 529 | 668 |
| 113 | `ESTADO-W1__is_missing` |  | 113 | 252 | 391 | 530 | 669 |
| 114 | `ESTADO-SDV-P__is_missing` |  | 114 | 253 | 392 | 531 | 670 |
| 115 | `ESTADO-DHSV__is_missing` |  | 115 | 254 | 393 | 532 | 671 |
| 116 | `ESTADO-M2__is_missing` |  | 116 | 255 | 394 | 533 | 672 |
| 117 | `ESTADO-W2__is_missing` |  | 117 | 256 | 395 | 534 | 673 |
| 118 | `ESTADO-SDV-GL__is_missing` |  | 118 | 257 | 396 | 535 | 674 |
| 119 | `ESTADO-XO__is_missing` |  | 119 | 258 | 397 | 536 | 675 |
| 120 | `ESTADO-PXO__is_missing` |  | 120 | 259 | 398 | 537 | 676 |
| 121 | `ABER-CKP__is_frozen` | quality · is_frozen | 121 | 260 | 399 | 538 | 677 |
| 122 | `P-MON-CKP__is_frozen` |  | 122 | 261 | 400 | 539 | 678 |
| 123 | `P-JUS-CKP__is_frozen` |  | 123 | 262 | 401 | 540 | 679 |
| 124 | `T-MON-CKP__is_frozen` |  | 124 | 263 | 402 | 541 | 680 |
| 125 | `T-JUS-CKP__is_frozen` |  | 125 | 264 | 403 | 542 | 681 |
| 126 | `P-MON-SDV-P__is_frozen` |  | 126 | 265 | 404 | 543 | 682 |
| 127 | `PT-P__is_frozen` |  | 127 | 266 | 405 | 544 | 683 |
| 128 | `P-TPT__is_frozen` |  | 128 | 267 | 406 | 545 | 684 |
| 129 | `T-TPT__is_frozen` |  | 129 | 268 | 407 | 546 | 685 |
| 130 | `P-PDG__is_frozen` |  | 130 | 269 | 408 | 547 | 686 |
| 131 | `T-PDG__is_frozen` |  | 131 | 270 | 409 | 548 | 687 |
| 132 | `ABER-CKGL__is_frozen` |  | 132 | 271 | 410 | 549 | 688 |
| 133 | `QGL__is_frozen` |  | 133 | 272 | 411 | 550 | 689 |
| 134 | `P-ANULAR__is_frozen` |  | 134 | 273 | 412 | 551 | 690 |
| 135 | `P-MON-CKGL__is_frozen` |  | 135 | 274 | 413 | 552 | 691 |
| 136 | `P-JUS-CKGL__is_frozen` |  | 136 | 275 | 414 | 553 | 692 |
| 137 | `QBS__is_frozen` |  | 137 | 276 | 415 | 554 | 693 |
| 138 | `P-JUS-BS__is_frozen` |  | 138 | 277 | 416 | 555 | 694 |

## Appendix — complete 695-name list (position → name)


### Block 0: `__mean`  (positions 0–138)

```
  0  dP_CKP__mean
  1  dP_CKGL__mean
  2  dT_CKP__mean
  3  dP_grad__mean
  4  dT_grad__mean
  5  dP_tree__mean
  6  dP_grad_roc__mean
  7  dP_CKP_roc__mean
  8  Cv_CKGL__mean
  9  cond_CKP__mean
 10  Cv_BS__mean
 11  hydrate_margin_prod__mean
 12  hydrate_ratio_prod__mean
 13  hydrate_margin_svc__mean
 14  P-PDG__dom_freq__mean
 15  P-PDG__spec_entropy__mean
 16  P-PDG__band_0.0_0.01__mean
 17  P-PDG__band_0.01_0.05__mean
 18  P-PDG__band_0.05_0.2__mean
 19  P-TPT__dom_freq__mean
 20  P-TPT__spec_entropy__mean
 21  P-TPT__band_0.0_0.01__mean
 22  P-TPT__band_0.01_0.05__mean
 23  P-TPT__band_0.05_0.2__mean
 24  PT-P__dom_freq__mean
 25  PT-P__spec_entropy__mean
 26  PT-P__band_0.0_0.01__mean
 27  PT-P__band_0.01_0.05__mean
 28  PT-P__band_0.05_0.2__mean
 29  QGL__dom_freq__mean
 30  QGL__spec_entropy__mean
 31  QGL__band_0.0_0.01__mean
 32  QGL__band_0.01_0.05__mean
 33  QGL__band_0.05_0.2__mean
 34  P-PDG__osc_amp__mean
 35  P-PDG__osc_std__mean
 36  P-PDG__osc_period__mean
 37  P-TPT__osc_amp__mean
 38  P-TPT__osc_std__mean
 39  P-TPT__osc_period__mean
 40  PT-P__osc_amp__mean
 41  PT-P__osc_std__mean
 42  PT-P__osc_period__mean
 43  QGL__osc_amp__mean
 44  QGL__osc_std__mean
 45  QGL__osc_period__mean
 46  P-MON-CKP__osc_amp__mean
 47  P-MON-CKP__osc_std__mean
 48  P-MON-CKP__osc_period__mean
 49  ESTADO-M1__delta__mean
 50  ESTADO-M1__changed__mean
 51  ESTADO-M1__time_since_change__mean
 52  ESTADO-W1__delta__mean
 53  ESTADO-W1__changed__mean
 54  ESTADO-W1__time_since_change__mean
 55  ESTADO-SDV-P__delta__mean
 56  ESTADO-SDV-P__changed__mean
 57  ESTADO-SDV-P__time_since_change__mean
 58  ESTADO-DHSV__delta__mean
 59  ESTADO-DHSV__changed__mean
 60  ESTADO-DHSV__time_since_change__mean
 61  ESTADO-M2__delta__mean
 62  ESTADO-M2__changed__mean
 63  ESTADO-M2__time_since_change__mean
 64  ESTADO-W2__delta__mean
 65  ESTADO-W2__changed__mean
 66  ESTADO-W2__time_since_change__mean
 67  ESTADO-SDV-GL__delta__mean
 68  ESTADO-SDV-GL__changed__mean
 69  ESTADO-SDV-GL__time_since_change__mean
 70  ESTADO-XO__delta__mean
 71  ESTADO-XO__changed__mean
 72  ESTADO-XO__time_since_change__mean
 73  ESTADO-PXO__delta__mean
 74  ESTADO-PXO__changed__mean
 75  ESTADO-PXO__time_since_change__mean
 76  ABER-CKP__norm__mean
 77  P-MON-CKP__norm__mean
 78  P-JUS-CKP__norm__mean
 79  T-MON-CKP__norm__mean
 80  T-JUS-CKP__norm__mean
 81  P-MON-SDV-P__norm__mean
 82  PT-P__norm__mean
 83  P-TPT__norm__mean
 84  T-TPT__norm__mean
 85  P-PDG__norm__mean
 86  T-PDG__norm__mean
 87  ABER-CKGL__norm__mean
 88  QGL__norm__mean
 89  P-ANULAR__norm__mean
 90  P-MON-CKGL__norm__mean
 91  P-JUS-CKGL__norm__mean
 92  QBS__norm__mean
 93  P-JUS-BS__norm__mean
 94  ABER-CKP__is_missing__mean
 95  P-MON-CKP__is_missing__mean
 96  P-JUS-CKP__is_missing__mean
 97  T-MON-CKP__is_missing__mean
 98  T-JUS-CKP__is_missing__mean
 99  P-MON-SDV-P__is_missing__mean
100  PT-P__is_missing__mean
101  P-TPT__is_missing__mean
102  T-TPT__is_missing__mean
103  P-PDG__is_missing__mean
104  T-PDG__is_missing__mean
105  ABER-CKGL__is_missing__mean
106  QGL__is_missing__mean
107  P-ANULAR__is_missing__mean
108  P-MON-CKGL__is_missing__mean
109  P-JUS-CKGL__is_missing__mean
110  QBS__is_missing__mean
111  P-JUS-BS__is_missing__mean
112  ESTADO-M1__is_missing__mean
113  ESTADO-W1__is_missing__mean
114  ESTADO-SDV-P__is_missing__mean
115  ESTADO-DHSV__is_missing__mean
116  ESTADO-M2__is_missing__mean
117  ESTADO-W2__is_missing__mean
118  ESTADO-SDV-GL__is_missing__mean
119  ESTADO-XO__is_missing__mean
120  ESTADO-PXO__is_missing__mean
121  ABER-CKP__is_frozen__mean
122  P-MON-CKP__is_frozen__mean
123  P-JUS-CKP__is_frozen__mean
124  T-MON-CKP__is_frozen__mean
125  T-JUS-CKP__is_frozen__mean
126  P-MON-SDV-P__is_frozen__mean
127  PT-P__is_frozen__mean
128  P-TPT__is_frozen__mean
129  T-TPT__is_frozen__mean
130  P-PDG__is_frozen__mean
131  T-PDG__is_frozen__mean
132  ABER-CKGL__is_frozen__mean
133  QGL__is_frozen__mean
134  P-ANULAR__is_frozen__mean
135  P-MON-CKGL__is_frozen__mean
136  P-JUS-CKGL__is_frozen__mean
137  QBS__is_frozen__mean
138  P-JUS-BS__is_frozen__mean
```

### Block 1: `__std`  (positions 139–277)

```
139  dP_CKP__std
140  dP_CKGL__std
141  dT_CKP__std
142  dP_grad__std
143  dT_grad__std
144  dP_tree__std
145  dP_grad_roc__std
146  dP_CKP_roc__std
147  Cv_CKGL__std
148  cond_CKP__std
149  Cv_BS__std
150  hydrate_margin_prod__std
151  hydrate_ratio_prod__std
152  hydrate_margin_svc__std
153  P-PDG__dom_freq__std
154  P-PDG__spec_entropy__std
155  P-PDG__band_0.0_0.01__std
156  P-PDG__band_0.01_0.05__std
157  P-PDG__band_0.05_0.2__std
158  P-TPT__dom_freq__std
159  P-TPT__spec_entropy__std
160  P-TPT__band_0.0_0.01__std
161  P-TPT__band_0.01_0.05__std
162  P-TPT__band_0.05_0.2__std
163  PT-P__dom_freq__std
164  PT-P__spec_entropy__std
165  PT-P__band_0.0_0.01__std
166  PT-P__band_0.01_0.05__std
167  PT-P__band_0.05_0.2__std
168  QGL__dom_freq__std
169  QGL__spec_entropy__std
170  QGL__band_0.0_0.01__std
171  QGL__band_0.01_0.05__std
172  QGL__band_0.05_0.2__std
173  P-PDG__osc_amp__std
174  P-PDG__osc_std__std
175  P-PDG__osc_period__std
176  P-TPT__osc_amp__std
177  P-TPT__osc_std__std
178  P-TPT__osc_period__std
179  PT-P__osc_amp__std
180  PT-P__osc_std__std
181  PT-P__osc_period__std
182  QGL__osc_amp__std
183  QGL__osc_std__std
184  QGL__osc_period__std
185  P-MON-CKP__osc_amp__std
186  P-MON-CKP__osc_std__std
187  P-MON-CKP__osc_period__std
188  ESTADO-M1__delta__std
189  ESTADO-M1__changed__std
190  ESTADO-M1__time_since_change__std
191  ESTADO-W1__delta__std
192  ESTADO-W1__changed__std
193  ESTADO-W1__time_since_change__std
194  ESTADO-SDV-P__delta__std
195  ESTADO-SDV-P__changed__std
196  ESTADO-SDV-P__time_since_change__std
197  ESTADO-DHSV__delta__std
198  ESTADO-DHSV__changed__std
199  ESTADO-DHSV__time_since_change__std
200  ESTADO-M2__delta__std
201  ESTADO-M2__changed__std
202  ESTADO-M2__time_since_change__std
203  ESTADO-W2__delta__std
204  ESTADO-W2__changed__std
205  ESTADO-W2__time_since_change__std
206  ESTADO-SDV-GL__delta__std
207  ESTADO-SDV-GL__changed__std
208  ESTADO-SDV-GL__time_since_change__std
209  ESTADO-XO__delta__std
210  ESTADO-XO__changed__std
211  ESTADO-XO__time_since_change__std
212  ESTADO-PXO__delta__std
213  ESTADO-PXO__changed__std
214  ESTADO-PXO__time_since_change__std
215  ABER-CKP__norm__std
216  P-MON-CKP__norm__std
217  P-JUS-CKP__norm__std
218  T-MON-CKP__norm__std
219  T-JUS-CKP__norm__std
220  P-MON-SDV-P__norm__std
221  PT-P__norm__std
222  P-TPT__norm__std
223  T-TPT__norm__std
224  P-PDG__norm__std
225  T-PDG__norm__std
226  ABER-CKGL__norm__std
227  QGL__norm__std
228  P-ANULAR__norm__std
229  P-MON-CKGL__norm__std
230  P-JUS-CKGL__norm__std
231  QBS__norm__std
232  P-JUS-BS__norm__std
233  ABER-CKP__is_missing__std
234  P-MON-CKP__is_missing__std
235  P-JUS-CKP__is_missing__std
236  T-MON-CKP__is_missing__std
237  T-JUS-CKP__is_missing__std
238  P-MON-SDV-P__is_missing__std
239  PT-P__is_missing__std
240  P-TPT__is_missing__std
241  T-TPT__is_missing__std
242  P-PDG__is_missing__std
243  T-PDG__is_missing__std
244  ABER-CKGL__is_missing__std
245  QGL__is_missing__std
246  P-ANULAR__is_missing__std
247  P-MON-CKGL__is_missing__std
248  P-JUS-CKGL__is_missing__std
249  QBS__is_missing__std
250  P-JUS-BS__is_missing__std
251  ESTADO-M1__is_missing__std
252  ESTADO-W1__is_missing__std
253  ESTADO-SDV-P__is_missing__std
254  ESTADO-DHSV__is_missing__std
255  ESTADO-M2__is_missing__std
256  ESTADO-W2__is_missing__std
257  ESTADO-SDV-GL__is_missing__std
258  ESTADO-XO__is_missing__std
259  ESTADO-PXO__is_missing__std
260  ABER-CKP__is_frozen__std
261  P-MON-CKP__is_frozen__std
262  P-JUS-CKP__is_frozen__std
263  T-MON-CKP__is_frozen__std
264  T-JUS-CKP__is_frozen__std
265  P-MON-SDV-P__is_frozen__std
266  PT-P__is_frozen__std
267  P-TPT__is_frozen__std
268  T-TPT__is_frozen__std
269  P-PDG__is_frozen__std
270  T-PDG__is_frozen__std
271  ABER-CKGL__is_frozen__std
272  QGL__is_frozen__std
273  P-ANULAR__is_frozen__std
274  P-MON-CKGL__is_frozen__std
275  P-JUS-CKGL__is_frozen__std
276  QBS__is_frozen__std
277  P-JUS-BS__is_frozen__std
```

### Block 2: `__min`  (positions 278–416)

```
278  dP_CKP__min
279  dP_CKGL__min
280  dT_CKP__min
281  dP_grad__min
282  dT_grad__min
283  dP_tree__min
284  dP_grad_roc__min
285  dP_CKP_roc__min
286  Cv_CKGL__min
287  cond_CKP__min
288  Cv_BS__min
289  hydrate_margin_prod__min
290  hydrate_ratio_prod__min
291  hydrate_margin_svc__min
292  P-PDG__dom_freq__min
293  P-PDG__spec_entropy__min
294  P-PDG__band_0.0_0.01__min
295  P-PDG__band_0.01_0.05__min
296  P-PDG__band_0.05_0.2__min
297  P-TPT__dom_freq__min
298  P-TPT__spec_entropy__min
299  P-TPT__band_0.0_0.01__min
300  P-TPT__band_0.01_0.05__min
301  P-TPT__band_0.05_0.2__min
302  PT-P__dom_freq__min
303  PT-P__spec_entropy__min
304  PT-P__band_0.0_0.01__min
305  PT-P__band_0.01_0.05__min
306  PT-P__band_0.05_0.2__min
307  QGL__dom_freq__min
308  QGL__spec_entropy__min
309  QGL__band_0.0_0.01__min
310  QGL__band_0.01_0.05__min
311  QGL__band_0.05_0.2__min
312  P-PDG__osc_amp__min
313  P-PDG__osc_std__min
314  P-PDG__osc_period__min
315  P-TPT__osc_amp__min
316  P-TPT__osc_std__min
317  P-TPT__osc_period__min
318  PT-P__osc_amp__min
319  PT-P__osc_std__min
320  PT-P__osc_period__min
321  QGL__osc_amp__min
322  QGL__osc_std__min
323  QGL__osc_period__min
324  P-MON-CKP__osc_amp__min
325  P-MON-CKP__osc_std__min
326  P-MON-CKP__osc_period__min
327  ESTADO-M1__delta__min
328  ESTADO-M1__changed__min
329  ESTADO-M1__time_since_change__min
330  ESTADO-W1__delta__min
331  ESTADO-W1__changed__min
332  ESTADO-W1__time_since_change__min
333  ESTADO-SDV-P__delta__min
334  ESTADO-SDV-P__changed__min
335  ESTADO-SDV-P__time_since_change__min
336  ESTADO-DHSV__delta__min
337  ESTADO-DHSV__changed__min
338  ESTADO-DHSV__time_since_change__min
339  ESTADO-M2__delta__min
340  ESTADO-M2__changed__min
341  ESTADO-M2__time_since_change__min
342  ESTADO-W2__delta__min
343  ESTADO-W2__changed__min
344  ESTADO-W2__time_since_change__min
345  ESTADO-SDV-GL__delta__min
346  ESTADO-SDV-GL__changed__min
347  ESTADO-SDV-GL__time_since_change__min
348  ESTADO-XO__delta__min
349  ESTADO-XO__changed__min
350  ESTADO-XO__time_since_change__min
351  ESTADO-PXO__delta__min
352  ESTADO-PXO__changed__min
353  ESTADO-PXO__time_since_change__min
354  ABER-CKP__norm__min
355  P-MON-CKP__norm__min
356  P-JUS-CKP__norm__min
357  T-MON-CKP__norm__min
358  T-JUS-CKP__norm__min
359  P-MON-SDV-P__norm__min
360  PT-P__norm__min
361  P-TPT__norm__min
362  T-TPT__norm__min
363  P-PDG__norm__min
364  T-PDG__norm__min
365  ABER-CKGL__norm__min
366  QGL__norm__min
367  P-ANULAR__norm__min
368  P-MON-CKGL__norm__min
369  P-JUS-CKGL__norm__min
370  QBS__norm__min
371  P-JUS-BS__norm__min
372  ABER-CKP__is_missing__min
373  P-MON-CKP__is_missing__min
374  P-JUS-CKP__is_missing__min
375  T-MON-CKP__is_missing__min
376  T-JUS-CKP__is_missing__min
377  P-MON-SDV-P__is_missing__min
378  PT-P__is_missing__min
379  P-TPT__is_missing__min
380  T-TPT__is_missing__min
381  P-PDG__is_missing__min
382  T-PDG__is_missing__min
383  ABER-CKGL__is_missing__min
384  QGL__is_missing__min
385  P-ANULAR__is_missing__min
386  P-MON-CKGL__is_missing__min
387  P-JUS-CKGL__is_missing__min
388  QBS__is_missing__min
389  P-JUS-BS__is_missing__min
390  ESTADO-M1__is_missing__min
391  ESTADO-W1__is_missing__min
392  ESTADO-SDV-P__is_missing__min
393  ESTADO-DHSV__is_missing__min
394  ESTADO-M2__is_missing__min
395  ESTADO-W2__is_missing__min
396  ESTADO-SDV-GL__is_missing__min
397  ESTADO-XO__is_missing__min
398  ESTADO-PXO__is_missing__min
399  ABER-CKP__is_frozen__min
400  P-MON-CKP__is_frozen__min
401  P-JUS-CKP__is_frozen__min
402  T-MON-CKP__is_frozen__min
403  T-JUS-CKP__is_frozen__min
404  P-MON-SDV-P__is_frozen__min
405  PT-P__is_frozen__min
406  P-TPT__is_frozen__min
407  T-TPT__is_frozen__min
408  P-PDG__is_frozen__min
409  T-PDG__is_frozen__min
410  ABER-CKGL__is_frozen__min
411  QGL__is_frozen__min
412  P-ANULAR__is_frozen__min
413  P-MON-CKGL__is_frozen__min
414  P-JUS-CKGL__is_frozen__min
415  QBS__is_frozen__min
416  P-JUS-BS__is_frozen__min
```

### Block 3: `__max`  (positions 417–555)

```
417  dP_CKP__max
418  dP_CKGL__max
419  dT_CKP__max
420  dP_grad__max
421  dT_grad__max
422  dP_tree__max
423  dP_grad_roc__max
424  dP_CKP_roc__max
425  Cv_CKGL__max
426  cond_CKP__max
427  Cv_BS__max
428  hydrate_margin_prod__max
429  hydrate_ratio_prod__max
430  hydrate_margin_svc__max
431  P-PDG__dom_freq__max
432  P-PDG__spec_entropy__max
433  P-PDG__band_0.0_0.01__max
434  P-PDG__band_0.01_0.05__max
435  P-PDG__band_0.05_0.2__max
436  P-TPT__dom_freq__max
437  P-TPT__spec_entropy__max
438  P-TPT__band_0.0_0.01__max
439  P-TPT__band_0.01_0.05__max
440  P-TPT__band_0.05_0.2__max
441  PT-P__dom_freq__max
442  PT-P__spec_entropy__max
443  PT-P__band_0.0_0.01__max
444  PT-P__band_0.01_0.05__max
445  PT-P__band_0.05_0.2__max
446  QGL__dom_freq__max
447  QGL__spec_entropy__max
448  QGL__band_0.0_0.01__max
449  QGL__band_0.01_0.05__max
450  QGL__band_0.05_0.2__max
451  P-PDG__osc_amp__max
452  P-PDG__osc_std__max
453  P-PDG__osc_period__max
454  P-TPT__osc_amp__max
455  P-TPT__osc_std__max
456  P-TPT__osc_period__max
457  PT-P__osc_amp__max
458  PT-P__osc_std__max
459  PT-P__osc_period__max
460  QGL__osc_amp__max
461  QGL__osc_std__max
462  QGL__osc_period__max
463  P-MON-CKP__osc_amp__max
464  P-MON-CKP__osc_std__max
465  P-MON-CKP__osc_period__max
466  ESTADO-M1__delta__max
467  ESTADO-M1__changed__max
468  ESTADO-M1__time_since_change__max
469  ESTADO-W1__delta__max
470  ESTADO-W1__changed__max
471  ESTADO-W1__time_since_change__max
472  ESTADO-SDV-P__delta__max
473  ESTADO-SDV-P__changed__max
474  ESTADO-SDV-P__time_since_change__max
475  ESTADO-DHSV__delta__max
476  ESTADO-DHSV__changed__max
477  ESTADO-DHSV__time_since_change__max
478  ESTADO-M2__delta__max
479  ESTADO-M2__changed__max
480  ESTADO-M2__time_since_change__max
481  ESTADO-W2__delta__max
482  ESTADO-W2__changed__max
483  ESTADO-W2__time_since_change__max
484  ESTADO-SDV-GL__delta__max
485  ESTADO-SDV-GL__changed__max
486  ESTADO-SDV-GL__time_since_change__max
487  ESTADO-XO__delta__max
488  ESTADO-XO__changed__max
489  ESTADO-XO__time_since_change__max
490  ESTADO-PXO__delta__max
491  ESTADO-PXO__changed__max
492  ESTADO-PXO__time_since_change__max
493  ABER-CKP__norm__max
494  P-MON-CKP__norm__max
495  P-JUS-CKP__norm__max
496  T-MON-CKP__norm__max
497  T-JUS-CKP__norm__max
498  P-MON-SDV-P__norm__max
499  PT-P__norm__max
500  P-TPT__norm__max
501  T-TPT__norm__max
502  P-PDG__norm__max
503  T-PDG__norm__max
504  ABER-CKGL__norm__max
505  QGL__norm__max
506  P-ANULAR__norm__max
507  P-MON-CKGL__norm__max
508  P-JUS-CKGL__norm__max
509  QBS__norm__max
510  P-JUS-BS__norm__max
511  ABER-CKP__is_missing__max
512  P-MON-CKP__is_missing__max
513  P-JUS-CKP__is_missing__max
514  T-MON-CKP__is_missing__max
515  T-JUS-CKP__is_missing__max
516  P-MON-SDV-P__is_missing__max
517  PT-P__is_missing__max
518  P-TPT__is_missing__max
519  T-TPT__is_missing__max
520  P-PDG__is_missing__max
521  T-PDG__is_missing__max
522  ABER-CKGL__is_missing__max
523  QGL__is_missing__max
524  P-ANULAR__is_missing__max
525  P-MON-CKGL__is_missing__max
526  P-JUS-CKGL__is_missing__max
527  QBS__is_missing__max
528  P-JUS-BS__is_missing__max
529  ESTADO-M1__is_missing__max
530  ESTADO-W1__is_missing__max
531  ESTADO-SDV-P__is_missing__max
532  ESTADO-DHSV__is_missing__max
533  ESTADO-M2__is_missing__max
534  ESTADO-W2__is_missing__max
535  ESTADO-SDV-GL__is_missing__max
536  ESTADO-XO__is_missing__max
537  ESTADO-PXO__is_missing__max
538  ABER-CKP__is_frozen__max
539  P-MON-CKP__is_frozen__max
540  P-JUS-CKP__is_frozen__max
541  T-MON-CKP__is_frozen__max
542  T-JUS-CKP__is_frozen__max
543  P-MON-SDV-P__is_frozen__max
544  PT-P__is_frozen__max
545  P-TPT__is_frozen__max
546  T-TPT__is_frozen__max
547  P-PDG__is_frozen__max
548  T-PDG__is_frozen__max
549  ABER-CKGL__is_frozen__max
550  QGL__is_frozen__max
551  P-ANULAR__is_frozen__max
552  P-MON-CKGL__is_frozen__max
553  P-JUS-CKGL__is_frozen__max
554  QBS__is_frozen__max
555  P-JUS-BS__is_frozen__max
```

### Block 4: `__last`  (positions 556–694)

```
556  dP_CKP__last
557  dP_CKGL__last
558  dT_CKP__last
559  dP_grad__last
560  dT_grad__last
561  dP_tree__last
562  dP_grad_roc__last
563  dP_CKP_roc__last
564  Cv_CKGL__last
565  cond_CKP__last
566  Cv_BS__last
567  hydrate_margin_prod__last
568  hydrate_ratio_prod__last
569  hydrate_margin_svc__last
570  P-PDG__dom_freq__last
571  P-PDG__spec_entropy__last
572  P-PDG__band_0.0_0.01__last
573  P-PDG__band_0.01_0.05__last
574  P-PDG__band_0.05_0.2__last
575  P-TPT__dom_freq__last
576  P-TPT__spec_entropy__last
577  P-TPT__band_0.0_0.01__last
578  P-TPT__band_0.01_0.05__last
579  P-TPT__band_0.05_0.2__last
580  PT-P__dom_freq__last
581  PT-P__spec_entropy__last
582  PT-P__band_0.0_0.01__last
583  PT-P__band_0.01_0.05__last
584  PT-P__band_0.05_0.2__last
585  QGL__dom_freq__last
586  QGL__spec_entropy__last
587  QGL__band_0.0_0.01__last
588  QGL__band_0.01_0.05__last
589  QGL__band_0.05_0.2__last
590  P-PDG__osc_amp__last
591  P-PDG__osc_std__last
592  P-PDG__osc_period__last
593  P-TPT__osc_amp__last
594  P-TPT__osc_std__last
595  P-TPT__osc_period__last
596  PT-P__osc_amp__last
597  PT-P__osc_std__last
598  PT-P__osc_period__last
599  QGL__osc_amp__last
600  QGL__osc_std__last
601  QGL__osc_period__last
602  P-MON-CKP__osc_amp__last
603  P-MON-CKP__osc_std__last
604  P-MON-CKP__osc_period__last
605  ESTADO-M1__delta__last
606  ESTADO-M1__changed__last
607  ESTADO-M1__time_since_change__last
608  ESTADO-W1__delta__last
609  ESTADO-W1__changed__last
610  ESTADO-W1__time_since_change__last
611  ESTADO-SDV-P__delta__last
612  ESTADO-SDV-P__changed__last
613  ESTADO-SDV-P__time_since_change__last
614  ESTADO-DHSV__delta__last
615  ESTADO-DHSV__changed__last
616  ESTADO-DHSV__time_since_change__last
617  ESTADO-M2__delta__last
618  ESTADO-M2__changed__last
619  ESTADO-M2__time_since_change__last
620  ESTADO-W2__delta__last
621  ESTADO-W2__changed__last
622  ESTADO-W2__time_since_change__last
623  ESTADO-SDV-GL__delta__last
624  ESTADO-SDV-GL__changed__last
625  ESTADO-SDV-GL__time_since_change__last
626  ESTADO-XO__delta__last
627  ESTADO-XO__changed__last
628  ESTADO-XO__time_since_change__last
629  ESTADO-PXO__delta__last
630  ESTADO-PXO__changed__last
631  ESTADO-PXO__time_since_change__last
632  ABER-CKP__norm__last
633  P-MON-CKP__norm__last
634  P-JUS-CKP__norm__last
635  T-MON-CKP__norm__last
636  T-JUS-CKP__norm__last
637  P-MON-SDV-P__norm__last
638  PT-P__norm__last
639  P-TPT__norm__last
640  T-TPT__norm__last
641  P-PDG__norm__last
642  T-PDG__norm__last
643  ABER-CKGL__norm__last
644  QGL__norm__last
645  P-ANULAR__norm__last
646  P-MON-CKGL__norm__last
647  P-JUS-CKGL__norm__last
648  QBS__norm__last
649  P-JUS-BS__norm__last
650  ABER-CKP__is_missing__last
651  P-MON-CKP__is_missing__last
652  P-JUS-CKP__is_missing__last
653  T-MON-CKP__is_missing__last
654  T-JUS-CKP__is_missing__last
655  P-MON-SDV-P__is_missing__last
656  PT-P__is_missing__last
657  P-TPT__is_missing__last
658  T-TPT__is_missing__last
659  P-PDG__is_missing__last
660  T-PDG__is_missing__last
661  ABER-CKGL__is_missing__last
662  QGL__is_missing__last
663  P-ANULAR__is_missing__last
664  P-MON-CKGL__is_missing__last
665  P-JUS-CKGL__is_missing__last
666  QBS__is_missing__last
667  P-JUS-BS__is_missing__last
668  ESTADO-M1__is_missing__last
669  ESTADO-W1__is_missing__last
670  ESTADO-SDV-P__is_missing__last
671  ESTADO-DHSV__is_missing__last
672  ESTADO-M2__is_missing__last
673  ESTADO-W2__is_missing__last
674  ESTADO-SDV-GL__is_missing__last
675  ESTADO-XO__is_missing__last
676  ESTADO-PXO__is_missing__last
677  ABER-CKP__is_frozen__last
678  P-MON-CKP__is_frozen__last
679  P-JUS-CKP__is_frozen__last
680  T-MON-CKP__is_frozen__last
681  T-JUS-CKP__is_frozen__last
682  P-MON-SDV-P__is_frozen__last
683  PT-P__is_frozen__last
684  P-TPT__is_frozen__last
685  T-TPT__is_frozen__last
686  P-PDG__is_frozen__last
687  T-PDG__is_frozen__last
688  ABER-CKGL__is_frozen__last
689  QGL__is_frozen__last
690  P-ANULAR__is_frozen__last
691  P-MON-CKGL__is_frozen__last
692  P-JUS-CKGL__is_frozen__last
693  QBS__is_frozen__last
694  P-JUS-BS__is_frozen__last
```
