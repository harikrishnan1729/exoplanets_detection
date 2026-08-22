# Exoplanet Detection from Kepler Light Curves

> Trying to find planets orbiting other stars — not by looking at them, but by noticing that a star got very slightly dimmer, very briefly, over and over again.

This repository is a hands-on, honest attempt at a genuinely hard machine learning problem: picking out the 0.7% of stars that have a planet from a pile of 5,000 brightness measurements, where "just say no to everything" scores 99.3% accuracy and finds absolutely nothing.

Everything here — the wins, the dead ends, the 717 automated training runs that mostly went nowhere — is documented below.

---

## Table of contents

1. [The science in plain language](#1-the-science-in-plain-language)
2. [The dataset](#2-the-dataset)
3. [The core difficulty: why this problem is genuinely hard](#3-the-core-difficulty-why-this-problem-is-genuinely-hard)
4. [Repository map](#4-repository-map)
5. [File-by-file walkthrough](#5-file-by-file-walkthrough)
   - [`src/download_data.py`](#51-srcdownload_datapy--getting-the-data)
   - [`src/explore_data.py`](#52-srcexplore_datapy--the-sanity-check)
   - [`src/dataset.py`](#53-srcdatasetpy--the-heart-of-the-project)
   - [`src/randomforest.py`](#54-srcrandomforestpy--the-current-model)
   - [`src/auto_train_loop.py`](#55-srcauto_train_looppy--the-robot-that-trains-all-night)
   - [`models/exoplanet_detector.keras`](#56-modelsexoplanet_detectorkeras--the-frozen-neural-network)
   - [`best_f1.txt` and `auto_train_log.txt`](#57-best_f1txt-and-auto_train_logtxt--the-scoreboard-and-the-diary)
   - [The ghosts: `train.py`, `evaluate.py`, `model.py`](#58-the-ghosts-trainpy-evaluatepy-modelpy)
6. [Every metric, explained properly](#6-every-metric-explained-properly)
7. [Results — what the numbers actually mean](#7-results--what-the-numbers-actually-mean)
8. [The full experiment history](#8-the-full-experiment-history)
9. [How to run this yourself](#9-how-to-run-this-yourself)
10. [Known bugs and rough edges](#10-known-bugs-and-rough-edges)
11. [Where this should go next](#11-where-this-should-go-next)

---

## 1. The science in plain language

### The transit method

You cannot see an exoplanet. A planet is a dark speck next to a nuclear furnace billions of times brighter, and it's all so far away that the whole system is a single pixel in a telescope.

So astronomers cheat. They don't look for the planet — they look for the **shadow**.

Imagine standing across a football field from a floodlight, and a mosquito flies in front of it. You'd never *see* the mosquito. But if you had a sensitive enough light meter, you'd measure a tiny dip in brightness as it passed. That's the transit method, and that's exactly what NASA's Kepler space telescope did for four years: it stared at ~150,000 stars and recorded how bright each one was, over and over, for years.

When a planet crosses in front of its star (a **transit**), the star's measured brightness drops. Then it comes back. Then, one orbital period later, it drops again.

```
brightness
    │
────┼──────╲    ╱─────────────────────╲    ╱──────────────
    │       ╲__╱                       ╲__╱
    │        ▲                          ▲
    │      transit 1                  transit 2
    └────────────────────────────────────────────────► time
             └────── one orbital period ──────┘
```

Three properties make a real transit recognisable:

| Property | What it looks like | Why it matters |
|---|---|---|
| **Periodic** | The dip repeats at a fixed interval | A one-off dip is a cosmic ray, an instrument glitch, or a passing asteroid |
| **U-shaped** | Flat-ish bottom, sharp shoulders | The planet fully covers a chunk of the disc, then slides off |
| **Shallow** | Typically 0.01%–1% of total brightness | A star-sized companion would produce a *huge* V-shaped dip instead |

That third row is the punchline for this project. **A Jupiter-sized planet in front of a Sun-sized star blocks about 1% of the light. An Earth-sized one blocks about 0.008%.** The signal we're hunting is buried under stellar flickering, instrument noise, and cosmic rays that are often *larger* than the thing we want to find.

### What that means for a machine learning model

The model's job is: *given ~3,200 brightness numbers taken over roughly 80 days, decide whether a faint repeating dip is hiding in there.*

It is a needle-in-a-haystack problem where the needle is also made of hay.

---

## 2. The dataset

**Source:** [Kepler Labelled Time Series Data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data) on Kaggle, derived from NASA's Kepler mission observations (Campaign 3).

Two CSV files:

| File | Rows (stars) | Columns | Confirmed planets | Non-planets |
|---|---|---|---|---|
| `exoTrain.csv` | 5,087 | 3,198 | **37** | 5,050 |
| `exoTest.csv` | 570 | 3,198 | **5** | 565 |

### The shape of a single row

```
LABEL, FLUX.1, FLUX.2, FLUX.3, ................................, FLUX.3197
  ▲     └──────────────────── 3,197 brightness measurements ──────────┘
  │                    one star, sampled over ~80 days
  │
  └─ 2 = has at least one confirmed exoplanet
     1 = no confirmed exoplanet
```

Each row is **one star's entire light curve** flattened into a row of a spreadsheet. Column `FLUX.1` is the earliest measurement, `FLUX.3197` the latest. The `LABEL` column is what we're trying to predict.

### The label shift, and why it exists

You'll see this line everywhere in the codebase:

```python
y_train = train.iloc[:, 0] - 1
```

The raw labels are `1` and `2`. Almost every ML library in existence assumes binary labels are `0` and `1` — `sklearn`'s `f1_score`, Keras' `sigmoid` output, `confusion_matrix` ordering, all of it. Subtracting 1 maps:

- `1` (no planet) → **`0`** — the *negative* class
- `2` (has planet) → **`1`** — the *positive* class

It's a one-character fix that prevents a whole category of silent, maddening bugs where your metrics look plausible but are computed against the wrong class.

---

## 3. The core difficulty: why this problem is genuinely hard

There are three separate problems stacked on top of each other, and each one has to be solved independently.

### Problem 1 — Extreme class imbalance (137:1)

There are 37 planets among 5,087 training stars. That's **0.73%**.

Machine learning models minimise a loss function. If 99.27% of your data is one class, the fastest, laziest way to make the loss go down is to predict that class every single time. The model isn't being stupid — it's doing exactly what you asked. You just asked the wrong question.

This actually happened, and it's preserved in the commit history:

```
commit 204ff54 — "confusion matrix [565 0; 5 0]"
```

Read that matrix:

```
                 predicted: no planet   predicted: planet
actual: no planet       565                     0
actual: planet            5                     0
```

The model predicted "no planet" for all 570 test stars. **Accuracy: 99.12%.** Planets found: **zero**. It is simultaneously one of the most accurate and most useless models you could build. That commit message is a monument to why accuracy is the wrong metric here.

### Problem 2 — The signal is weaker than the noise

Even the strongest transits in this data are fractions of a percent. Stars are not stable light bulbs; they have flares, starspots rotating in and out of view, and pulsations. The telescope itself contributes thermal drift and pointing jitter. A lot of what looks like a "dip" is just a star being a star.

### Problem 3 — Time is not aligned across stars

This one is subtle and it's the biggest limitation of the current approach.

Consider two stars, both with planets:

- Star A's transits happen at time indices 200, 900, 1600, 2300
- Star B's transits happen at time indices 50, 410, 770, 1130

Column `FLUX.200` means "the 200th measurement of this particular star." It does **not** mean the same physical thing for Star A as it does for Star B. There is no reason for a transit to land in the same column across different stars — the orbital periods differ, and the observation start times are arbitrary relative to the orbits.

Any model that treats each column as an independent feature — which is exactly what a fully-connected neural network and a Random Forest both do — is trying to learn a rule like *"if column 200 is low, it's a planet."* That rule cannot generalise, because column 200 is meaningless.

> **This is the single biggest reason performance plateaus around F1 ≈ 0.57.** It is not a hyperparameter problem. It's a representation problem. See [section 11](#11-where-this-should-go-next) for the fix.

---

## 4. Repository map

```
exoplanets_detection/                 (default branch: randomforest)
│
├── src/
│   ├── download_data.py       ← 3 lines. Pulls the dataset from Kaggle.
│   ├── explore_data.py        ← 5 lines. Prints labels to confirm the -1 shift works.
│   ├── dataset.py             ← THE IMPORTANT ONE. All loading + preprocessing.
│   ├── randomforest.py        ← The current, active model. Train → predict → report.
│   ├── auto_train_loop.py     ← Automated overnight retrain-until-it-improves loop.
│   └── __pycache__/           ← Compiled bytecode (should be gitignored — it isn't)
│       ├── dataset.cpython-313.pyc
│       ├── model.cpython-313.pyc      ← evidence of the deleted MLP
│       └── model_cnn.cpython-313.pyc  ← evidence of a deleted CNN variant
│
├── models/
│   └── exoplanet_detector.keras  ← 10 MB. The trained neural network, frozen at its best.
│
├── auto_train_log.txt         ← 5,077 lines. Every automated run, timestamped.
├── best_f1.txt                ← One number: 0.571429. The high score.
├── .gitignore                 ← Currently empty.
└── README.md                  ← (this file)
```

**A note on Python version:** the `__pycache__` files are tagged `cpython-313`, so this was developed on **Python 3.13**.

---

## 5. File-by-file walkthrough

### 5.1 `src/download_data.py` — getting the data

```python
import kagglehub
path = kagglehub.dataset_download("keplersmachines/kepler-labelled-time-series-data")
print(path)
```

Three lines, and they earn their place. `kagglehub` downloads the dataset, unzips it, caches it locally, and returns the path. Run it once. It prints something like:

```
C:\Users\HARIKRISHNAN\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3
```

**You need to copy that printed path into `dataset.py`.** That's the manual handshake between these two files, and it's the first thing that will trip up anyone cloning this repo.

Why the data isn't committed: `exoTrain.csv` is roughly 250 MB. GitHub warns at 50 MB and hard-rejects at 100 MB. Downloading on demand is the right call.

---

### 5.2 `src/explore_data.py` — the sanity check

```python
import pandas as pd

path = r"C:\Users\HARIKRISHNAN\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3"
train = pd.read_csv(f"{path}/exoTrain.csv")
print(f" {train.iloc[:, 0] - 1}")
```

This is a five-line file whose entire purpose is to answer one question: *"did the `-1` actually do what I think it did?"*

It prints the label column after shifting, and you eyeball it to confirm you see `0`s and `1`s rather than `1`s and `2`s.

This is not throwaway code — it's **the correct instinct**. Before building anything on top of an assumption, verify the assumption in isolation. Most debugging nightmares are caused by skipping this step.

*(Note the `r"..."` prefix — a raw string. Without it, Python reads `\U` in `C:\Users` as the start of a Unicode escape sequence and throws a `SyntaxError`. Windows paths and Python string escapes are old enemies.)*

---

### 5.3 `src/dataset.py` — the heart of the project

This is the most consequential file in the repository. Every model — the MLP, the CNN, the Random Forest — consumes its output. If something is wrong here, it is wrong everywhere.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from scipy.signal import savgol_filter

dataset_path = r"dataset_path"          # ← YOU MUST EDIT THIS
```

> ⚠️ **Setup step:** `dataset_path` is a literal placeholder string. Replace it with the path printed by `download_data.py` before anything will run.

#### Step A — Smoothing the light curves

```python
def smooth_light_curves(x):
    return savgol_filter(
        x,
        window_length=11,
        polyorder=5,
        axis=1
    )
```

**What a Savitzky-Golay filter actually does.** Most smoothers (a moving average, say) work by replacing each point with the average of its neighbours. That kills noise but it also flattens peaks and fills in valleys — which is a catastrophe here, because *the valley is the entire signal we're looking for*.

Savitzky-Golay is smarter. For each point, it slides a window over the data, fits a **polynomial** through the points in that window by least squares, and replaces the centre point with the polynomial's value there. Because a polynomial can curve, the filter preserves the shape of dips and peaks while still removing point-to-point jitter.

The two knobs:

- `window_length=11` — look at 11 consecutive measurements at a time (5 on either side of the centre point). Must be odd.
- `polyorder=5` — fit a 5th-degree polynomial to those 11 points. Must be less than `window_length`.

A 5th-degree curve through only 11 points is very flexible — it hugs the data closely. **So this is deliberately gentle smoothing.** It scrubs off the highest-frequency crackle and leaves essentially everything else intact. That's the right conservative choice when your signal might be only 0.01% deep: you'd rather keep some noise than accidentally erase a real transit.

`axis=1` means "smooth along the time direction, within each star's row" — never across different stars, which would be nonsense.

Visually:

```
raw:      ▁▂▁▃▂▁▂▁▃▁▂▁▁▂▁▂█▇█▂▁▃▁▂▁▃▂▁▂▁▃▁     ← jittery, dip present
smoothed: ▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂█▇█▂▂▂▂▂▂▂▂▂▂▂▂▂     ← jitter gone, dip preserved
                          └── the transit ──┘
```

#### Step B — Loading and splitting off the labels

```python
def load_data():
    train = pd.read_csv(f"{dataset_path}\exoTrain.csv")
    test  = pd.read_csv(f"{dataset_path}/exoTest.csv")

    x_test  = test.iloc[:, 1:]      # all flux columns
    y_test  = test.iloc[:, 0] - 1   # label, shifted to 0/1

    x_train = train.iloc[:, 1:]
    y_train = train.iloc[:, 0] - 1
```

`iloc[:, 1:]` = every row, columns 1 onward (the 3,197 flux values). `iloc[:, 0]` = every row, column 0 only (the label).

*(Minor inconsistency worth noting: line 21 uses a backslash separator and line 22 a forward slash. Both happen to work on Windows; the backslash version will break on Linux/macOS. See [known bugs](#10-known-bugs-and-rough-edges).)*

#### Step C — Per-star normalisation, and why the `StandardScaler` got commented out

This is the most thoughtful decision in the file, and it's easy to miss because the *rejected* approach is still sitting there in comments:

```python
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(x_train)
    # x_test_scaled = scaler.transform(x_test)

    x_scaled = (x_train - x_train.mean(axis=1, keepdims=True)) / (
        x_train.std(axis=1, keepdims=True) + 1e-8
    )
    x_test_scaled = (x_test - x_test.mean(axis=1, keepdims=True)) / (
        x_test.std(axis=1, keepdims=True) + 1e-8
    )
```

**The difference is the axis, and it changes everything.**

`StandardScaler` normalises **column-wise** (`axis=0`). It would compute the mean and standard deviation of `FLUX.1` *across all 5,087 stars* and standardise that column. But as established in [Problem 3](#problem-3--time-is-not-aligned-across-stars), `FLUX.1` isn't a coherent quantity across stars — different stars have wildly different intrinsic brightnesses, and time index 1 corresponds to a different orbital phase for each of them. Column-wise statistics mix apples, oranges, and red giants.

The code here normalises **row-wise** (`axis=1`): each star is standardised *against itself*.

```
For each star:
    subtract that star's own average brightness
    divide by that star's own variability
```

After this, every light curve is centred on 0 with a spread of 1, and the question the model faces changes from *"is this star bright?"* to *"does this star dip relative to its own normal behaviour?"* — which is precisely the physics.

Three bonus properties, all of which matter:

1. **Intrinsic brightness is removed.** A dim star and a bright star with identical transit shapes now look identical to the model.
2. **No data leakage, by construction.** Because each row's statistics come only from that row, no information ever flows from the training set into the test set. A column-wise scaler fitted on train and applied to test is *correct* but fragile; row-wise normalisation makes the mistake impossible.
3. **`+ 1e-8`** guards against division by zero for a hypothetical perfectly flat light curve. Cheap insurance against a `NaN` that would silently poison everything downstream.

#### Step D — The stratified train/validation split

```python
    X_train, X_val, y_train, y_val = train_test_split(
        x_scaled, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
```

`stratify=y_train` is doing critical work here. With only 37 positive examples in 5,087 rows, a naive random 80/20 split could easily hand the validation set 1 planet, or 12, or 0 — pure luck. Stratification forces the split to preserve the class ratio in both halves.

The resulting sizes:

| Split | Stars | Planets |
|---|---|---|
| Training | 4,069 | 30 |
| Validation | 1,018 | 7 |
| Test (separate file, untouched) | 570 | 5 |

*(These aren't guesses — they're confirmed by the commit message `"random forest test 1 with confusion 26 FA 4T 3M 985"`, whose four numbers sum to exactly 1,018.)*

`random_state=42` fixes the shuffle so the split is identical on every run. Without it, you'd never know whether a score changed because your model improved or because the dice rolled differently.

#### Step E — The two commented-out blocks (paths not taken)

```python
    ''' use when expecting a CNN'''
    # X_train = X_train.reshape(-1, X_train.shape[1], 1)
    # X_val = X_val.reshape(-1, X_val.shape[1], 1)
    # x_test_scaled = x_test_scaled.reshape(-1, x_test_scaled.shape[1], 1)
```

A 1D convolutional network expects input shaped `(batch, timesteps, channels)`, not `(batch, features)`. These three lines add the trailing channel dimension: `(4069, 3197)` → `(4069, 3197, 1)`. Uncomment them when swapping the CNN back in. The `-1` tells NumPy "figure out the batch size yourself."

```python
    # ros = RandomOverSampler(random_state=42)
    # X_train, y_train = ros.fit_resample(X_train, y_train)
```

**`RandomOverSampler`** attacks the imbalance by duplicating minority-class rows until the classes are balanced — here it would copy the 30 training planets ~135 times each, producing 4,039 planet rows to sit alongside 4,039 non-planet rows.

It's disabled, and that's a defensible call. With only 30 distinct positives, duplicating them 135× doesn't create new information — it creates 135 identical chances for the model to memorise each specific light curve. Overfitting becomes almost guaranteed. The Random Forest gets the same effect more safely via `class_weight="balanced"` (see below), which reweights the *loss* rather than the *data*.

The original README also mentions trying **SMOTE** (Synthetic Minority Over-sampling Technique), which is the more sophisticated cousin: instead of copying a minority point, it picks two nearby minority points and invents a new synthetic example somewhere on the line between them. In a 3,197-dimensional space with 30 real examples, "nearby" is not a well-defined concept — the nearest neighbour of a planet is essentially random. The synthetic light curves it produces are blends of unrelated stars, and they don't correspond to any physical object. It was tried; it didn't stick.

#### What comes out

```python
    return X_train, X_val, x_test_scaled, y_train, y_val, y_test
```

Six objects, in that exact order. Every consumer unpacks them positionally, so the order is load-bearing.

---

### 5.4 `src/randomforest.py` — the current model

This is what the default branch is named after, and it's the approach that ended up working best.

```python
from dataset import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV

x_train, x_val, x_test, y_train, y_val, y_test = load_data()

print("Training samples:", x_train.shape[0])
print("Validation samples:", x_val.shape[0])
print("Test samples:", x_test.shape[0])
```

Expected output:

```
Training samples: 4069
Validation samples: 1018
Test samples: 570
```

#### The forest itself

```python
model = RandomForestClassifier(
    n_estimators=320,
    random_state=100,
    n_jobs=-1,
    class_weight="balanced",
    min_samples_split=2,
    min_samples_leaf=2,
    max_depth=None
)
```

**What a Random Forest is, from the ground up.**

Start with one **decision tree**. A tree learns a nested series of yes/no questions:

```
                  Is FLUX.1204 < -2.1 ?
                    /              \
                  yes               no
                  /                  \
      Is FLUX.887 < -1.8 ?        [ no planet ]
          /          \
        yes           no
        /              \
  [ planet ]      [ no planet ]
```

A single tree is fast and interpretable but notoriously unstable — perturb the training data slightly and you get a completely different tree. It memorises.

A **forest** fixes this by building many trees and having them vote. Two sources of deliberate randomness keep the trees from being clones:

1. **Bagging** — each tree trains on a bootstrap sample (a random draw *with replacement* of 4,069 rows from the 4,069 available, so roughly 63% unique rows, some repeated).
2. **Feature subsampling** — at every split, the tree may only consider a random subset of the features (by default `sqrt(3197) ≈ 57` of them).

Individually each tree is mediocre and biased in its own idiosyncratic direction. Averaged, the idiosyncrasies cancel and what remains is signal. This is the wisdom-of-crowds effect, made rigorous.

The parameters, one at a time:

| Parameter | Value | What it does |
|---|---|---|
| `n_estimators` | `320` | Build 320 trees. More trees = smoother, more stable probabilities, never worse accuracy — just slower. 320 is comfortably past the point of diminishing returns. |
| `random_state` | `100` | Fixes all the internal randomness so the run is reproducible. |
| `n_jobs` | `-1` | Use every CPU core. Trees are independent, so this parallelises almost perfectly. |
| `class_weight` | `"balanced"` | **The imbalance fix.** See below. |
| `min_samples_split` | `2` | A node may split if it holds ≥2 samples — i.e. split as deep as possible. |
| `min_samples_leaf` | `2` | A leaf must contain ≥2 samples. Mild regularisation; blocks leaves built around a single memorised star. |
| `max_depth` | `None` | No depth ceiling. Grow until leaves are pure or hit `min_samples_leaf`. |

**`class_weight="balanced"` deserves its own paragraph.** It tells sklearn to weight each class inversely to its frequency:

```
weight(class) = n_samples / (n_classes × n_samples_in_class)

weight(no planet) = 4069 / (2 × 4039) ≈ 0.50
weight(planet)    = 4069 / (2 ×   30) ≈ 67.8
```

Every planet now counts as much as ~136 non-planets when the trees evaluate a split. In effect, misclassifying a single planet hurts the model as much as misclassifying 136 ordinary stars. This is the elegant version of oversampling: you get the rebalanced incentives without physically duplicating rows and without inviting memorisation.

#### Predicting probabilities, not labels

```python
model.fit(x_train, y_train)
y_prob = model.predict_proba(x_test)[:, 1]
```

`.predict()` would return hard `0`/`1` answers using an implicit 50% cutoff. `.predict_proba()` returns the full probability for both classes as an `(n_samples, 2)` array; `[:, 1]` grabs column 1 — the probability of the *planet* class.

For a Random Forest, that probability has a beautifully concrete meaning: **it's the fraction of the 320 trees that voted "planet."** A score of 0.05 means 16 trees out of 320 raised their hand.

Keeping the probability instead of the label is what makes the next step possible.

#### The threshold — the most important line in the file

```python
threshold = 0.05
y_test_pred = (y_prob >= threshold).astype(int)
```

A 5% cutoff sounds absurd until you internalise what it means: **"if even 1 in 20 trees thinks this star has a planet, flag it for a human to look at."**

This is not the model being reckless. It's an explicit statement about the *cost of the two kinds of mistake*, and in astronomy those costs are wildly asymmetric:

- **A false positive** costs a few minutes of an astronomer's follow-up time. Annoying. Cheap.
- **A false negative** means a real planet is never discovered. Irreversible. Expensive.

At the default 0.50 threshold this model finds almost nothing — the 30 planets in training are simply not enough for a majority of trees to ever become confident. Dropping to 0.05 converts the classifier from *"tell me what you're sure about"* into *"tell me what's worth a second look."* That's exactly the right job description for a real candidate-screening pipeline, where this model would be stage one and a human or a physics-based fitting routine would be stage two.

> **This is the "threshold problem" the original README mentions.** With so few positives, the model's confidence swings wildly between retrainings; there is no threshold that is stable and correct. Choosing 0.05 is less a tuned hyperparameter than a policy decision about which errors you're willing to make.

#### The report

```python
print("\nTest results:")
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))
```

`zero_division=0` prevents a crash-adjacent warning: if the model predicts zero positives, precision is `0/0`. This tells sklearn to print `0.00` instead of complaining.

Typical output shape (illustrative numbers):

```
Test results:
[[560   5]
 [  1   4]]

              precision    recall  f1-score   support

           0     0.9982    0.9912    0.9947       565
           1     0.4444    0.8000    0.5714         5

    accuracy                         0.9895       570
```

Read the class-`1` row and ignore everything else. That row is the entire point of the project.

#### The commented-out sections

Roughly half the file is commented out, and it's a fair record of the investigation:

- A **`RandomizedSearchCV`** block (removed in commit `33fba2f`) that sampled 20 random combinations from a grid of `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`, and `max_features`, scored by `f1` with 3-fold cross-validation. The surviving hyperparameters are its descendants.
- A **threshold sweep** over `[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`, printing a confusion matrix at each — the empirical work behind picking 0.05.
- A block printing `y_prob[y_test == 1]` — *the model's probability for each star that genuinely has a planet.* This is the single most diagnostic thing you can print in an imbalanced problem. If the real planets score 0.03, 0.06, 0.11, 0.02, 0.09 while the noise sits at 0.001, the model **has** learned something and you just need a lower threshold. If the real planets are scattered indistinguishably among the noise, no threshold will save you and you need better features. Highly recommended to uncomment.

*(`RandomizedSearchCV` and `f1_score` are still imported but no longer used. Harmless, but they're leftovers.)*

---

### 5.5 `src/auto_train_loop.py` — the robot that trains all night

This file is the most unusual thing in the repository, and honestly the most interesting. It's not a model — it's **infrastructure**. It automates the tedious human loop of *"train, check the score, is it better? if not, train again"* and it commits and pushes to GitHub whenever a new record is set.

The docstring states the design plainly:

```
1. Run evaluate.py, capture its stdout, parse the line "f1: <score>".
2. Compare against the best F1 seen so far (stored in best_f1.txt).
3. If it's a NEW BEST: save it, git add / commit (message includes the score) / push.
4. If NOT better: run train.py, evaluate again, repeat until F1 "saturates".
```

#### Configuration

```python
PROJECT_DIR = Path(r"c:\Users\HARIKRISHNAN\Desktop\exoplanet dectetor")
PYTHON_EXE  = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
EVALUATE_SCRIPT = PROJECT_DIR / "src" / "evaluate.py"
TRAIN_SCRIPT    = PROJECT_DIR / "src" / "train.py"

BEST_F1_FILE = PROJECT_DIR / "best_f1.txt"
LOG_FILE     = PROJECT_DIR / "auto_train_log.txt"
```

Hardcoded Windows paths, invoking the virtualenv's `python.exe` directly so the subprocess inherits the right packages regardless of how the parent was launched. (`(dectetor)` — the typo is in the actual directory name.)

```python
MAX_ITERATIONS = 50
PATIENCE = 100
MIN_IMPROVEMENT = 0.001
CONTINUE_AFTER_BEST = True
```

- `MAX_ITERATIONS` — hard cap so it can't loop forever.
- `PATIENCE` — how many non-improving rounds before giving up. **Set to 100, which exceeds `MAX_ITERATIONS`, so it can never actually trigger.** Earlier log entries show it at `4`, which did work; someone raised it and disabled it by accident.
- `MIN_IMPROVEMENT` — improvements smaller than 0.001 don't count as progress, so noise doesn't reset the patience counter.
- `CONTINUE_AFTER_BEST` — keep hunting after setting a record, rather than stopping.

#### Running a subprocess and watching it live

```python
def run_and_capture(cmd, cwd):
    process = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines = []
    for line in process.stdout:
        print(line, end="")          # live output
        captured_lines.append(line)
    process.wait()
    ...
    return "".join(captured_lines)
```

This is a nicely-built helper. The design choices:

- `stderr=subprocess.STDOUT` merges error output into the same stream, so nothing gets lost.
- `text=True` decodes bytes to `str` automatically.
- `bufsize=1` requests line buffering, so output appears as it's produced rather than in a lump at the end.
- Iterating `for line in process.stdout` **both prints and collects** — you can watch training progress in real time *and* still have the full text to parse afterwards. Most naive implementations force you to pick one.

#### Extracting the score with a regex

```python
F1_PATTERN = re.compile(r"f1\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def parse_f1(output: str):
    matches = F1_PATTERN.findall(output)
    if not matches:
        return None
    return float(matches[-1])
```

The pattern, decoded piece by piece:

| Fragment | Meaning |
|---|---|
| `f1` | the literal characters `f1` |
| `\s*` | any amount of whitespace (or none) |
| `:` | a colon |
| `\s*` | more optional whitespace |
| `([0-9]*\.?[0-9]+)` | **capture group**: optional digits, optional dot, then at least one digit |
| `re.IGNORECASE` | matches `f1:`, `F1:`, `F1 : `, all of them |

So `f1: 0.5714`, `F1:0.5714`, and `F1 : .5714` all parse correctly. Taking `matches[-1]` (the *last* match) is deliberate — if the evaluation script prints intermediate F1 values during a threshold sweep, only the final summary line counts.

#### Persisting the record

```python
def get_best_f1():
    if BEST_F1_FILE.exists():
        try:
            return float(BEST_F1_FILE.read_text().strip())
        except ValueError:
            return -1.0
    return -1.0
```

Reads `best_f1.txt`. Returns `-1.0` if the file is missing or corrupt — a sentinel that's below any legal F1 score, so the very first real evaluation automatically becomes the record. The `try/except` means a truncated or garbage file degrades gracefully instead of crashing the overnight run.

#### The git automation

```python
def commit_and_push_best(f1_value: float):
    message = f"f1: {f1_value:.6f}"
    if not git(["add", "."], PROJECT_DIR):
        return False
    if not git(["commit", "-m", message], PROJECT_DIR):
        log("Nothing to commit or commit failed - continuing anyway.")
    return git(["push", "origin", "main"], PROJECT_DIR)
```

This is why the commit history reads like a scoreboard:

```
e7bbe5f  f1: 0.5
05767ca  f1: 0.444444
16cff23  f1: 0.333333
69301f9  f1: 0.285714
a9dec77  f1: 0.250000
```

Every one of those was written by a machine, at the moment it beat its own record, and pushed to GitHub automatically. The version history *is* the experiment log. That's a genuinely good idea.

*(It pushes to `origin main`, but the default branch is now `randomforest` — this will fail until updated.)*

#### The Karplus-Strong easter egg

```python
def karplus(frequency):
    sample_rate = 44100
    delay = int(sample_rate / frequency)
    buffer = np.random.uniform(-1, 1, delay)
    samples = []
    for _ in range(44100):
        samples.append(buffer[0])
        first  = buffer[0]
        second = buffer[1]
        new_sample = 0.996 * (first + second) / 2
        buffer = np.append(buffer[1:], new_sample)
    ...
```

This has nothing to do with exoplanets, and it's delightful. It's the **Karplus-Strong algorithm** — a famously elegant piece of 1980s physical-modelling synthesis that produces a startlingly realistic plucked-string sound from almost nothing. It's called from `git()` so the machine goes *plink* every time it commits a new best model: an audible notification that something good happened while you were asleep.

How it works, because it's worth knowing:

1. **Fill a buffer with noise.** Buffer length = `sample_rate / frequency`. For 124 Hz that's 355 samples. Pure white noise contains every frequency at once — this is the "pluck," the instant of chaos when a string is first displaced.
2. **Loop the buffer, averaging as you go.** Output the first sample, then compute the average of the first two, and push that average onto the back of the buffer.
3. **The averaging is a low-pass filter.** High frequencies are exactly what averaging destroys fastest. So with each pass around the loop, the noise loses its brightness — while the loop length preserves the fundamental pitch.
4. **The `0.996` factor** bleeds a tiny amount of energy on each pass, so the note decays to silence instead of ringing forever.

The result is a note that starts bright and percussive and mellows into a pure tone as it fades — which is precisely what a real plucked string does, because a real string also loses its high harmonics to friction faster than its fundamental. Ten lines of code, real physics.

`samples / np.max(np.abs(samples))` normalises to the range [-1, 1], `* 32767` scales to 16-bit integer range, and `np.column_stack((audio, audio))` duplicates the mono signal into two identical channels for stereo playback.

> ⚠️ **It's currently broken.** The final lines reference `p.sndarray.make_sound(...)` and `p.time.wait(2000)`, but `p` is never imported — presumably `import pygame as p`. As written, calling `karplus()` raises `NameError`, and since `git()` calls it unconditionally, **the git automation will crash.** It worked before the sound was added (the log proves it), and it was added in commit `5bf7e09` on 2026-08-11 — after the last logged run. [Fix in section 10.](#10-known-bugs-and-rough-edges)

#### The main loop

```python
for iteration in range(1, MAX_ITERATIONS + 1):
    f1 = run_evaluate()
    if f1 is None:
        sys.exit(1)

    if f1 > best_f1 + MIN_IMPROVEMENT or best_f1 < 0:
        best_f1 = f1
        save_best_f1(best_f1)
        no_improve_streak = 0
        commit_and_push_best(best_f1)
        if not CONTINUE_AFTER_BEST:
            return
    else:
        no_improve_streak += 1
        if no_improve_streak >= PATIENCE:
            return

    run_train()
```

Straightforward and correct: evaluate → compare → record or increment the streak → retrain → repeat.

Why retraining at all produces different results, given no hyperparameters change: the neural network's weight initialisation, dropout masks, and batch shuffling are all random. With only 30 positive training examples, that randomness dominates. Each retrain is effectively a lottery ticket, and this loop buys 717 of them.

> ⚠️ **The methodological catch.** This loop selects the best model *by its score on the test set*, and the test set is the thing that's supposed to be untouched until the very end. Run 717 lotteries and keep the winner, and the winning score is partly measuring luck rather than skill. The honest fix is to select on the *validation* set (which `dataset.py` already provides and nothing currently uses) and touch the test set exactly once. **F1 = 0.571 should therefore be read as an optimistic ceiling, not an unbiased estimate.** Flagging this isn't a criticism of the project — it's the kind of thing that's genuinely easy to miss and worth understanding deeply.

---

### 5.6 `models/exoplanet_detector.keras` — the frozen neural network

10 MB, saved by Keras 3.15.0 on **2026-08-11 at 21:03:38** — timestamped to the exact second the F1 hit its record of 0.571429. This file *is* the champion.

A `.keras` file is a zip archive. Unpacked, it contains `config.json` (architecture + compilation settings), `model.weights.h5` (the learned parameters), and `metadata.json`. Reading the config back out gives the exact architecture:

```
Input                (None, 3197)
  ↓
Dense    256 units,  ReLU
Dropout  rate 0.3
  ↓
Dense    128 units,  ReLU
Dropout  rate 0.3
  ↓
Dense     64 units,  ReLU
Dropout  rate 0.3
  ↓
Dense     32 units,  ReLU
  ↓
Dense      1 unit,   Sigmoid   → probability of planet
```

Compiled with:

```
optimizer : Adam (learning_rate = 0.001, β₁ = 0.9, β₂ = 0.999)
loss      : BinaryFocalCrossentropy (gamma = 2.0, alpha = 0.25,
                                     apply_class_balancing = False)
metrics   : accuracy, precision, recall, AUC
```

Roughly 900,000 parameters, the overwhelming majority in that first `3197 × 256` weight matrix — which is exactly the layer trying to learn per-time-index rules that don't generalise. That's where the 10 MB goes.

**The design decisions, explained:**

**Funnel architecture (256 → 128 → 64 → 32 → 1).** Each layer compresses the representation. The first layer looks for local patterns across the raw time series; each subsequent layer combines the previous layer's findings into fewer, more abstract features; the final single neuron condenses everything into one number. Narrowing forces the network to discard information, which is a form of regularisation — it can't just pass the input through unchanged.

**ReLU** (`max(0, x)`) is the standard hidden activation. It's cheap, and critically it doesn't saturate for positive inputs, so gradients flow back through many layers without vanishing.

**Sigmoid** on the output squashes any real number into (0, 1), giving something interpretable as a probability.

**Dropout(0.3)** randomly zeroes 30% of a layer's outputs on every training step. This sounds destructive and is essential. It stops the network from relying on any single neuron ("neuron #47 always fires for planets"), because neuron #47 might be switched off. Every neuron must contribute something independently useful. With 30 positive training examples and ~900k parameters, the network could trivially memorise every planet; dropout is a large part of what prevents that. It's active only during training — at inference time all neurons participate.

**`BinaryFocalCrossentropy` is the most sophisticated choice here and deserves real explanation.**

Standard binary cross-entropy loss for a single example is `-log(p_correct)`. If the model says 0.9 for a true positive, loss = 0.105. If it says 0.99, loss = 0.010. Small, but not zero — and multiplied across 4,039 easy negatives that the model already gets right with 99% confidence, those small losses sum to something that dominates the total. **The gradient signal from the 30 hard, interesting planets gets drowned out by 4,039 easy stars the model already understands.**

Focal loss (Lin et al., 2017, originally invented for dense object detection — a problem with the same pathology) multiplies the loss by a modulating factor:

```
FL(p) = -(1 - p)^γ · log(p)          with γ = 2.0
```

Look at what `(1 - p)^2` does:

| Model's confidence in the correct answer | Modulating factor | Effect |
|---|---|---|
| p = 0.99 (easy, already right) | (0.01)² = **0.0001** | loss cut by 10,000× — effectively silenced |
| p = 0.90 (fairly confident) | (0.10)² = **0.01** | loss cut by 100× |
| p = 0.50 (genuinely uncertain) | (0.50)² = **0.25** | loss cut by 4× |
| p = 0.10 (badly wrong) | (0.90)² = **0.81** | loss almost untouched |

Easy examples are muted; hard examples keep shouting. The model is forced to spend its learning capacity on the cases it doesn't yet understand — which, in this dataset, are the planets.

`apply_class_balancing=False` means the additional `alpha` class-weighting term is switched off, so focal loss is doing the imbalance work on its own via the modulating factor alone. (The stored `alpha=0.25` is Keras' default and is inert when balancing is disabled.)

**`Adam`** (Adaptive Moment Estimation) adapts the learning rate per parameter using running estimates of the gradient's mean and variance. Parameters with consistently small gradients get larger effective steps. It's the sensible default for exactly this reason: it works well without much tuning.

**The metrics list is itself a statement of intent.** Tracking `precision`, `recall`, and `AUC` alongside `accuracy` means the accuracy number was never being trusted on its own — which, given [Problem 1](#problem-1--extreme-class-imbalance-1371), is the correct posture.

---

### 5.7 `best_f1.txt` and `auto_train_log.txt` — the scoreboard and the diary

**`best_f1.txt`** contains exactly one thing:

```
0.571429
```

That's it. Written by `save_best_f1()` with `f"{value:.6f}"`. It's the persistent memory that lets you stop the automation and restart it days later without losing the record.

**`auto_train_log.txt`** is 5,077 lines covering **2026-08-07 19:43** through **2026-08-11 21:03** — five days of near-continuous automated experimentation.

A single successful record-break, in the log's own words:

```
[2026-08-07 19:43:08] Starting. Current best F1 on record: -1.0
[2026-08-07 19:43:08] --- Iteration 1/50 ---
[2026-08-07 19:43:08] Running: ...\python.exe ...\src\evaluate.py
[2026-08-07 19:43:16] Current F1 = 0.250000 | Best so far = -1.000000
[2026-08-07 19:43:16] New best F1! (0.250000, improvement 0.250000)
[2026-08-07 19:43:16] git add . -> rc=0
[2026-08-07 19:43:17] git commit -m f1: 0.250000 -> rc=0
[2026-08-07 19:43:22] git push origin main -> rc=0
[2026-08-07 19:43:22] Committed and pushed best F1 = 0.250000
```

And the final entry — the moment the record was set and never beaten:

```
[2026-08-11 21:03:46] Current F1 = 0.571429 | Best so far = 0.500000
[2026-08-11 21:03:46] New best F1! (0.571429, improvement 0.071429)
```

The `log()` function is deliberately dual-purpose:

```python
def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)                                    # to the terminal, live
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")                       # to disk, permanent
```

Open in append mode `"a"` so nothing is ever overwritten across runs, with explicit UTF-8 encoding so Windows' default encoding doesn't mangle anything.

---

### 5.8 The ghosts: `train.py`, `evaluate.py`, `model.py`

These files **no longer exist on the current branch** — they were deleted in commits `e7f3051`, `3ee7fa9`, and `94b80ff`. But `auto_train_loop.py` still calls them by name, and `__pycache__` still holds their compiled bytecode. They're documented here because the loop is unrunnable without them, and because they explain how `exoplanet_detector.keras` came to exist.

They're recoverable with `git show 33fba2f^:src/train.py` and similar.

#### `src/model.py` — the architecture factory

```python
import tensorflow as tf

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0,
                                                     apply_class_balancing=False),
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")]
    )
    return model
```

This matches the saved `.keras` file byte for byte in structure. It's the source of the frozen model.

#### The CNN that was tried and abandoned

Three earlier commits (`815e4ac`, `b16e549`, `204ff54`, around 2026-07-23 to 07-26) contain a completely different `model.py` — a 1D convolutional network:

```python
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation="relu"),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0,
                                                     apply_class_balancing=True),
        metrics=["accuracy"]
    )
    return model
```

**A CNN is theoretically the right tool for this problem, and it's worth understanding why.**

A `Conv1D` layer slides a small learnable window (here 5 timesteps wide) along the entire light curve, applying the same weights at every position. This gives it **translation invariance**: if it learns to recognise the shape of a transit dip, it recognises that shape *anywhere in the sequence*. That directly attacks [Problem 3](#problem-3--time-is-not-aligned-across-stars) — the transit no longer has to be in the same column across stars.

The layer stack:

- **`Conv1D(32, kernel_size=5)`** — 32 different 5-sample pattern detectors. Early filters typically learn primitives: edges, small dips, upward slopes.
- **`Conv1D(64, kernel_size=5)`** — 64 detectors operating on the first layer's output, so they see combinations: "a downward edge followed by a flat stretch followed by an upward edge" — i.e. the shape of a transit.
- **`BatchNormalization`** — renormalises each layer's activations to zero mean and unit variance per mini-batch. Keeps the distribution of inputs to the next layer stable as weights change during training, which lets you train faster and more reliably.
- **`MaxPooling1D(pool_size=2)`** — halves the sequence length by keeping the maximum of each adjacent pair. Cuts computation and adds small-shift tolerance.
- **`GlobalAveragePooling1D`** — collapses the entire remaining sequence to a single value per filter by averaging. This is what makes the network accept variable-position signals; it asks *"did this filter fire anywhere?"* rather than *"did it fire at position 200?"*
- **`apply_class_balancing=True`** — unlike the MLP, this version *did* enable focal loss's alpha weighting.

The commit that followed this one is titled **`"its all worse now"`**, and the CNN was reverted to the MLP. Which is an entirely believable outcome: with only 30 positive examples, a CNN — despite being architecturally correct — has nowhere near enough data to learn what a transit looks like from scratch. The right architecture with insufficient data still loses.

*(The `model_cnn.cpython-313.pyc` in `__pycache__` suggests a separate CNN module also existed locally at some point, never committed as source.)*

#### `src/train.py` — the training driver

```python
from dataset import load_data
from model import create_model
import tensorflow as tf

x_train, x_val, x_test, y_train, y_val, y_test = load_data()

model = create_model(x_train.shape[1:])
print(model.summary())

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=20,
    batch_size=20,
    verbose=2,
    callbacks=[early_stopping]
)

model.save("models/exoplanet_detector.keras")
```

`x_train.shape[1:]` gives `(3197,)` — the shape of one sample, dropping the batch dimension.

**`EarlyStopping` is the most important thing here.** It watches validation loss after every epoch. If validation loss fails to improve for 5 consecutive epochs (`patience=5`), training halts.

`restore_best_weights=True` is the part people forget: without it, you stop training but keep the *final* — already-degrading — weights. With it, Keras rewinds the model to the epoch that scored best on validation. You get the best model, not the last one.

Why this matters so much here: with 30 positive examples and 900k parameters, the network can memorise the training set completely. Training loss will keep falling long after the model has stopped learning anything real. Validation loss is the tripwire — the moment it turns upward, the model has switched from learning to memorising.

`batch_size=20` is small, and that's a considered choice. With planets at 0.73% frequency, a batch of 20 contains a planet only about 14% of the time — but small batches mean many more gradient updates per epoch, so those planet-containing batches arrive frequently enough to matter. Larger batches would average each planet's signal away among its neighbours.

A commented-out `class_weight={0: 1, 1: 2}` shows an earlier, gentler attempt at imbalance handling that focal loss eventually replaced.

#### `src/evaluate.py` — scoring and threshold search

```python
_, _, X_test, _, _, y_test = load_data()
model = tf.keras.models.load_model("models/exoplanet_detector.keras")
y_pred_prob = model.predict(X_test, verbose=0).flatten()

thresholds = np.arange(0.01, 1.00, 0.01)
best_threshold = 0
best_f1 = 0

for t in thresholds:
    y_pred = (y_pred_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred = (y_pred_prob >= 0.112578).astype(int)      # ← hardcoded, ignores the sweep
print(f"Threshold = {best_threshold:.2f} f1: {best_f1}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))
```

The underscores in the unpacking are Python convention for "I know six things come back; I only want these two."

`.flatten()` converts Keras' `(570, 1)` output into a flat `(570,)` array so it lines up with `y_test`.

The loop tries all 99 thresholds from 0.01 to 0.99 and keeps whichever maximises F1. **This is the line `auto_train_loop.py` greps for** — the `f1: {best_f1}` in that print statement is what the regex `f1\s*:\s*(...)` captures.

It also prints probability statistics and, valuably, the model's probability for every genuine planet in the test set:

```python
planet_indices = y_test[y_test == 1].index
for idx in planet_indices:
    print(f"Sample {idx}: Probability = {y_pred_prob[idx]:.6f}")
```

That listing tells you at a glance whether the model is close-but-miscalibrated or genuinely lost.

> ⚠️ **Two real bugs here.** First, the confusion matrix and classification report are computed at a **hardcoded `0.112578`**, not at `best_threshold` — so the printed matrix does not correspond to the printed F1. Second and more seriously, the threshold sweep optimises against `y_test`, which means the reported F1 is the *best possible* score on the test set, chosen with knowledge of the answers. Both are fixable; see below.

---

## 6. Every metric, explained properly

### The confusion matrix

Every binary prediction lands in exactly one of four buckets:

```
                        PREDICTED
                  no planet      planet
              ┌─────────────┬─────────────┐
   no planet  │     TN      │     FP      │
              │ True Neg.   │ False Pos.  │  ← a false alarm
   ACTUAL     ├─────────────┼─────────────┤
      planet  │     FN      │     TP      │
              │ False Neg.  │ True Pos.   │
              │  ← a MISS   │  ← a find!  │
              └─────────────┴─────────────┘
```

Everything else is arithmetic on these four numbers.

In this project the two error types have completely different real-world costs:

- **False Positive (FP)** — you flag an ordinary star. An astronomer looks, finds nothing, moves on. Cost: a little time.
- **False Negative (FN)** — a real planet is filed as boring and never examined again. Cost: an undiscovered world.

That asymmetry is the entire justification for the aggressive 0.05 threshold.

### Accuracy — and why it lies here

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

*"What fraction of all predictions were correct?"*

For the all-negative model: `(0 + 565) / 570 = 99.12%`.

99.12% accurate. Zero planets found. **Accuracy is not a metric for imbalanced problems; it's a way of not noticing that your model does nothing.**

### Precision

```
Precision = TP / (TP + FP)
```

*"Of the stars I flagged as planets, what fraction really were?"*

This is the **trustworthiness** of a positive prediction. Precision of 0.44 means: when this model raises the alarm, it's right a bit less than half the time. Low precision wastes follow-up effort.

### Recall (a.k.a. sensitivity, true positive rate)

```
Recall = TP / (TP + FN)
```

*"Of all the planets that actually exist, what fraction did I find?"*

This is **coverage**. Recall of 0.80 means: 4 of the 5 real planets were caught; 1 slipped through.

For a discovery survey, recall is arguably the metric that matters most. A planet you never flag is a planet nobody ever looks at.

### The tension between them

Precision and recall pull against each other, and the threshold is the dial between them:

```
threshold → 0.01     threshold → 0.50      threshold → 0.99
"flag everything"    "be balanced"         "be certain"

recall    ▲▲▲▲▲      recall    ▲▲▲         recall    ▲
precision ▲          precision ▲▲▲         precision ▲▲▲▲▲

You find every         middle ground        You're almost
planet, plus 500                            never wrong, and
false alarms                                you find nothing
```

You can trivially max out either one alone. Recall 1.00: predict "planet" for everything. Precision 1.00: predict "planet" for exactly the one star you're most sure about. Neither is useful. Which is why we need:

### F1-score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is the **harmonic mean** of precision and recall, and the choice of *harmonic* rather than arithmetic mean is the whole point.

Consider precision = 1.0, recall = 0.0 (you flagged one star, correctly, and missed everything else):

- Arithmetic mean: `(1.0 + 0.0) / 2 = 0.50` — looks respectable, which is a lie.
- Harmonic mean: `2 × (1.0 × 0.0) / (1.0 + 0.0) = 0.00` — correctly identifies this as worthless.

**The harmonic mean is dominated by the smaller value.** You cannot score well on F1 by being excellent at one thing and terrible at the other. You have to be decent at both. That's why it's the metric this project optimises, and why `best_f1.txt` exists at all.

Equivalent formulation, sometimes more intuitive:

```
F1 = 2·TP / (2·TP + FP + FN)
```

### AUC (Area Under the ROC Curve)

Tracked as a Keras metric during training. AUC has an elegant interpretation: **pick one random planet and one random non-planet; AUC is the probability the model assigns a higher score to the planet.**

0.5 = coin flip. 1.0 = perfect ranking.

Its virtue is that it's **threshold-independent** — it measures whether the model *ranks* planets above non-planets, regardless of where you eventually draw the line. Given how unstable the threshold is in this project, AUC is arguably the fairest measure of whether the model has learned anything real.

---

## 7. Results — what the numbers actually mean

### The headline

| | |
|---|---|
| **Best F1-score achieved** | **0.571429** |
| Achieved by | The MLP (`exoplanet_detector.keras`) |
| Achieved on | 2026-08-11 at 21:03:38 |
| After | 717 automated evaluations across 36 sessions |

### Decoding 0.571429

That number is exactly **4/7**, which is not a coincidence. Using `F1 = 2·TP / (2·TP + FP + FN)` with the test set's 5 planets, the combination that produces it is:

```
TP = 4     ← found 4 of the 5 real planets
FN = 1     ← missed 1
FP = 5     ← flagged 5 ordinary stars by mistake

F1 = 2(4) / (2(4) + 5 + 1) = 8/14 = 0.5714 ✓

Precision = 4/9  = 0.444    Recall = 4/5 = 0.800
```

**In human terms:** *given 570 stars, the model handed back a shortlist of 9. Four of them were genuine planets. It missed one. An astronomer following up that shortlist would find 4 of the 5 planets in the field after examining 9 objects instead of 570 — a 63× reduction in work, at the cost of missing 20% of the targets.*

That is not a solved problem. It is also, unambiguously, **a model that has learned something real.** Random guessing at that flag rate would find a planet essentially never.

### Compare against the baseline

| Model | Accuracy | Planets found | Precision | Recall | F1 |
|---|---|---|---|---|---|
| "Always say no planet" | **99.12%** | 0 of 5 | 0.00 | 0.00 | **0.00** |
| This project's best | 98.95% | **4 of 5** | 0.44 | 0.80 | **0.571** |

The best model is *less accurate* than the do-nothing baseline. It is also infinitely more useful. If you needed one table to explain why accuracy is the wrong metric for imbalanced problems, this is it.

### Evidence from the commit messages

The auto-loop's commit titles double as a results log:

| Commit | Message | Interpretation |
|---|---|---|
| `89e8a63` | `confusion matrix [999 12:6 1]` | Early run: 1 planet found, 12 false alarms, 6 missed |
| `204ff54` | `confusion matrix [565 0; 5 0]` | **The collapse** — predicted "no planet" 570/570 times |
| `815e4ac` | `its all worse now` | The CNN experiment, reverted |
| `c31a3ea` | `best ive seen so far with MLP` | Return to the fully-connected network |
| `09c618f` | `random forest test 1 with confusion 26 FA 4T 3M 985` | RF on validation: 4 found, 3 missed, 26 false alarms (of 1,018) |
| `5bf7e09` | `model with 4 planets found with 87 false alarms` | Very low threshold — high recall, precision destroyed |
| `34962f1` | `4 with 49 FA` | Threshold raised — false alarms nearly halved |
| `fa13c1a` | `19FA with 4planets` | Tuned further — still 4 planets, only 19 false alarms |

That last sequence — 87 → 49 → 19 false alarms while holding 4 planets found — is the threshold tuning working exactly as intended, and it's the clearest demonstration in the whole repo of *why* keeping probabilities instead of hard labels was the right call.

### The distribution of 717 attempts

Every F1 the automation ever recorded, tallied:

| F1 score | Times observed |
|---|---|
| 0.250000 | 66 |
| 0.222222 | 64 |
| 0.200000 | 55 |
| 0.181818 | 54 |
| 0.285714 | 48 |
| 0.166667 | 42 |
| 0.153846 | 42 |
| 0.133333 | 37 |
| 0.142857 | 36 |
| 0.333333 | 27 |
| ... | ... |
| **0.571429** | **1** |

**The typical retrain lands around F1 ≈ 0.20.** The record is roughly *three times* the median outcome, and it happened exactly once in 717 tries. That is the clearest possible statement that run-to-run variance dominates this problem — and the strongest argument for the caveat in [section 5.5](#55-srcauto_train_looppy--the-robot-that-trains-all-night) about the record being an optimistic ceiling rather than a reliable expectation.

### The climb

Six records were set over five days:

```
F1
0.6 │                                                          ● 0.571429
    │                                                   ● 0.500
0.5 │
    │                                            ● 0.444444
0.4 │
    │                                     ● 0.333333
0.3 │                       ● 0.285714
    │        ● 0.250000
0.2 │
    └─────────────────────────────────────────────────────────────────►
     Aug 07                                                    Aug 11
```

*(A confusion matrix of `[[504, 27], [6, 1]]` appears in the original README from a SMOTE experiment. Its four values sum to 538, which doesn't match either the 1,018-row validation set or the 570-row test set, so it presumably came from a differently-configured split during an earlier iteration.)*

---

## 8. The full experiment history

Reading the commit log chronologically gives an honest picture of how this was actually built — including the parts that didn't work.

| Date | Milestone |
|---|---|
| **2026-07-17** | `hi` — repository created |
| **2026-07-19** | First working MLP. Confusion `[999 12; 6 1]` — one planet found |
| **2026-07-23** | `last changes before CNN` — pivot to convolutions |
| **2026-07-25** | The collapse: `[565 0; 5 0]`, model predicts all-negative |
| **2026-07-26** | `improved` — recovering |
| **~2026-07** | `its all worse now` — CNN abandoned, back to the MLP |
| **~2026-08** | `best ive seen so far with MLP` — focal loss + row-wise normalisation land |
| **2026-08-07** | `auto_train_loop.py` written. Automation begins. F1 0.25 → 0.286 → 0.333 |
| **2026-08-11** | F1 0.444 → 0.500 → **0.571429**. Karplus-Strong plink added |
| **2026-08-12** | Pivot to Random Forest. `randomforest` branch becomes default. Dead files deleted |

The arc is worth noticing: **preprocessing → architecture → loss function → automation → algorithm change.** Each stage was tried properly, evaluated, and either kept or abandoned on evidence. The `randomforest` branch being the default is the current verdict — trees, with `class_weight="balanced"` and an aggressive threshold, turned out to be the more reliable performer on this much data.

---

## 9. How to run this yourself

### Prerequisites

```bash
python --version    # 3.13 recommended (matches the committed bytecode)
```

### Install

```bash
git clone https://github.com/harikrishnan1729/exoplanets_detection.git
cd exoplanets_detection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install pandas numpy scipy scikit-learn imbalanced-learn kagglehub matplotlib
# only needed if you want the neural network path:
pip install tensorflow
```

*(There is no `requirements.txt` on the current branch — one existed earlier in history but was empty.)*

### 1. Download the data

```bash
python src/download_data.py
```

Copy the path it prints.

### 2. Point `dataset.py` at the data

Open `src/dataset.py` and replace the placeholder:

```python
dataset_path = r"dataset_path"                    # ← before
dataset_path = r"C:\Users\YOU\.cache\kagglehub\datasets\keplersmachines\kepler-labelled-time-series-data\versions\3"   # ← after
```

While you're there, fix the separator inconsistency on line 21 (`\exoTrain.csv` → `/exoTrain.csv`) so it also runs on macOS and Linux.

### 3. Verify

```bash
cd src
python explore_data.py
```

You should see a column of `0`s and `1`s. If you see `1`s and `2`s, the label shift isn't working.

### 4. Train and evaluate

```bash
cd src          # must run from src/ — randomforest.py does `from dataset import load_data`
python randomforest.py
```

First run takes a couple of minutes (parsing a 250 MB CSV is the slow part). Expect:

```
Training samples: 4069
Validation samples: 1018
Test samples: 570

Test results:
[[...  ...]
 [...  ...]]
              precision    recall  f1-score   support
           0      ...        ...       ...        565
           1      ...        ...       ...          5
```

**Look at the class-`1` row.** That's the result.

### 5. (Optional) Experiment with the threshold

Uncomment the sweep block at the bottom of `randomforest.py`, or just edit `threshold = 0.05` and re-run. Watch precision and recall trade against each other. This is the most instructive five minutes you can spend with this repo.

---

## 10. Known bugs and rough edges

Documented honestly, with fixes.

| # | Location | Issue | Fix |
|---|---|---|---|
| 1 | `dataset.py:9` | `dataset_path = r"dataset_path"` is a placeholder — nothing runs until it's set | Set it, or read from an env var / CLI arg |
| 2 | `dataset.py:21` | `f"{dataset_path}\exoTrain.csv"` uses a backslash while line 22 uses `/` — breaks on macOS/Linux | Use `Path(dataset_path) / "exoTrain.csv"` for both |
| 3 | `auto_train_loop.py:78-80` | `karplus()` calls `p.sndarray.make_sound(...)` but `p` is never imported → `NameError` on every git call | Add `import pygame as p` + `p.mixer.init()`, or wrap the call in `try/except` |
| 4 | `auto_train_loop.py:72` | `np.append(buffer[1:], ...)` inside a 44,100-iteration loop reallocates the array every time — O(n²) | Use `collections.deque(maxlen=delay)` |
| 5 | `auto_train_loop.py:144` | Pushes to `origin main`, but the default branch is now `randomforest` | Change the branch name, or detect it with `git rev-parse --abbrev-ref HEAD` |
| 6 | `auto_train_loop.py:32-33` | Points at `src/evaluate.py` and `src/train.py`, both deleted from the branch | Restore them (`git show 33fba2f^:src/train.py > src/train.py`) or retarget at `randomforest.py` |
| 7 | `auto_train_loop.py:43` | `PATIENCE = 100` exceeds `MAX_ITERATIONS = 50`, so the saturation check can never fire | Set `PATIENCE` well below `MAX_ITERATIONS` (it was `4` in earlier runs) |
| 8 | `auto_train_loop.py` (design) | Selects the best model by **test-set** F1 across 717 runs → optimistically biased | Select on the validation set; evaluate on test exactly once, at the end |
| 9 | `evaluate.py:35` | Confusion matrix computed at hardcoded `0.112578` rather than the `best_threshold` just found | `y_pred = (y_pred_prob >= best_threshold).astype(int)` |
| 10 | `evaluate.py:18-30` | Threshold sweep optimises against `y_test` | Sweep on validation, apply the chosen threshold to test |
| 11 | `.gitignore` | Empty, so `__pycache__/` and a 10 MB `.keras` file are tracked | Add `__pycache__/`, `*.pyc`, `.venv/`, `.ipynb_checkpoints/`, `*.csv` |
| 12 | Repo-wide | No `requirements.txt` on the current branch | `pip freeze > requirements.txt` |
| 13 | `dataset.py`, `randomforest.py` | `StandardScaler`, `RandomOverSampler`, `RandomizedSearchCV`, `f1_score` imported but unused | Remove, or keep with a comment noting they're for the commented-out paths |
| 14 | Repo-wide | Hardcoded Windows absolute paths in three files | Move to a `config.py` or environment variables |

---

## 11. Where this should go next

Ordered by expected impact.

### 1. Fix the representation (this is the big one)

Everything else is a rounding error next to this. The models currently see raw, unaligned time series where column *k* means nothing consistent across stars. The standard fixes, in order of ambition:

**Phase folding.** Run a Box Least Squares (BLS) periodogram — a classical algorithm designed for exactly this — to find the most likely orbital period. Then wrap the light curve on that period so every transit stacks on top of every other transit. A 0.01% dip repeated 12 times becomes a single, unmistakable, 12×-reinforced dip. This alone typically transforms the problem. `astropy.timeseries.BoxLeastSquares` implements it.

**Global + local views.** The approach that made Shallue & Vanderburg's 2018 network work (it found Kepler-90i, an eighth planet in a known system): feed the network *two* inputs — a phase-folded view of the whole orbit, and a zoomed-in view centred on the transit itself. One provides context, the other provides detail.

**Frequency-domain features.** An FFT of the light curve converts "a dip every 340 timesteps" into a sharp peak at a specific frequency — position-independent by construction.

**Hand-crafted physical features.** Depth, duration, ingress/egress slope, period, transit shape (U-shaped for planets, V-shaped for grazing binaries), odd-vs-even transit depth (a difference indicates an eclipsing binary, not a planet). Thirty engineered features that actually mean something will beat 3,197 features that don't, especially with only 37 positives.

### 2. Cross-validation instead of a single split

Seven validation planets is far too few to measure anything reliably. Use `StratifiedKFold` with 5 folds so every planet serves as a test case exactly once, and report mean ± standard deviation. Given that the 717-run F1 distribution ranges from 0.087 to 0.571, error bars aren't optional — they're the only honest way to report a result.

### 3. Clean up the evaluation protocol

Select thresholds and models on validation; touch the test set once. This will lower the reported number and make it trustworthy.

### 4. Try gradient boosting

`XGBoost` or `LightGBM` with `scale_pos_weight=135` frequently outperform Random Forests on tabular problems, and both have first-class support for imbalanced data.

### 5. Reframe as anomaly detection

With 37 positives against 5,050 negatives, this might be better posed as *"learn what a normal star looks like, then flag whatever doesn't fit."* An autoencoder trained only on non-planet light curves would reconstruct ordinary stars well and planet-hosting stars badly; reconstruction error becomes the anomaly score. This uses all 5,050 negatives productively instead of treating them as the boring majority class.

### 6. Report precision-recall curves, not point estimates

A single F1 at a single threshold hides the whole trade-off. Plot precision against recall across all thresholds and report **average precision** — the right summary statistic for a heavily imbalanced problem, and far more informative than any one number.

### 7. Housekeeping

Add `requirements.txt` and a real `.gitignore`, move hardcoded paths into config, restore or retire the ghost files, and consider `git lfs` for the model checkpoint.

---

## Built with

| Tool | Role here |
|---|---|
| **Python 3.13** | Everything |
| **pandas** | Reading and slicing the 250 MB CSVs |
| **NumPy** | Array maths, row-wise normalisation, the Karplus-Strong synth |
| **SciPy** | `savgol_filter` for light-curve smoothing |
| **scikit-learn** | Random Forest, stratified splitting, all metrics |
| **TensorFlow / Keras 3.15** | The MLP and the abandoned CNN |
| **imbalanced-learn** | `RandomOverSampler`, SMOTE (both explored, both disabled) |
| **Matplotlib** | Light-curve plotting during exploration |
| **kagglehub** | Dataset download |
| **Git / GitHub** | Version control — *and*, via the auto-loop, the experiment log itself |

---

## Closing note

The most valuable thing in this repository isn't the 0.571 F1. It's the trail: a commit literally titled `its all worse now`, a confusion matrix commit that documents the model doing nothing at all, a `.gitignore` that never got filled in, and an automated training loop that plays a synthesised guitar note when it beats its own record.

The project takes a genuinely hard problem seriously — it identifies the class imbalance, rejects accuracy as a metric, reaches for focal loss and class weighting and per-star normalisation, tries a CNN because a CNN *should* work, and when it doesn't, says so and moves on. Then it builds a robot to run the experiment 717 times overnight.

The ceiling it hit is a representation ceiling, not an effort ceiling. Phase-fold the light curves and this same code will go considerably further.
