# Experiment 1: Summary

A short update on the first full run: what we built, how we tested it, and what we found.

Notebooks: `build_dataset.ipynb` → `train_model.ipynb`. Saved model and results are in `models/`.

---

## 1. What we are trying to do

A rider opens the Bike Share Toronto app and sees "1 e-bike" at a station 10 minutes away. Will that e-bike still be there when they arrive?

The app only shows what is there **right now**. It doesn't predict anything. Our model tries to answer:

> "If I get to station S in **h** minutes (2 to 15), will there be at least one **classic bike**, at least one **e-bike**, and at least one **free dock**?"

The model gives a **chance** for each of the three (for example "e-bike: 72%"), not just yes or no.

### Hard cases and easy cases

Each label (classic, e-bike, dock) is checked on its own:

- **Hard case:** the station shows **0, 1 or 2** right now. The answer could go either way, so **the model answers**.
- **Easy case:** the station shows **3 or more**. It is almost always still there a few minutes later, so a simple rule says **"yes"**.

All results below are for **hard cases only**.

---

## 2. The data

- **Where it comes from:** station snapshots from Bike Share Toronto (how many classic bikes, e-bikes and free docks each station has), plus hourly weather from Environment Canada.
- **Size:** about 11.4 million example queries across 1,073 stations.
- **One example** = one station + one moment in the past + one arrival time `h`. The inputs are only what we knew at that moment. The answer is what the station actually had `h` minutes later.
- **Split by date**, so the model is always tested on days it has never seen:

| Part | Dates (2026, Toronto time) | Examples | Used for |
|---|---|---|---|
| Train | Sep 29 – Oct 18 | 7.5 M | Learning |
| Validation | Oct 19 – Oct 23 | 2.1 M | Tuning and picking settings |
| Test | Oct 24 – Oct 28 | 1.7 M | Final score, looked at once |

---

## 3. The models

We compare our neural network against the app and two simpler models. All of them give a chance for each label.

| Model | How it works |
|---|---|
| **App rule** | 100% if the count right now is at least 1, otherwise 0%. This is what riders rely on today. |
| **Lookup table** | Groups past cases by count now, arrival time and recent trend (going up, flat or down). The chance is how often the item was there in that group. |
| **GBM (gradient-boosted trees)** | A few hundred small decision trees, each fixing the errors of the ones before. A strong standard model for table data. |
| **MLP (our neural network)** | Described below. |

### Our MLP

**What goes in (52 numbers):**
- Counts of classic bikes, e-bikes and free docks: now, 10 minutes ago and 30 minutes ago
- How those counts are changing (going up or down)
- Arrival time `h`, time of day, weekday or weekend
- Station size, location, temperature, humidity
- A small learned "fingerprint" of 16 numbers for each station, so the model can learn each station's habits

**The layers:**

```
52 inputs → 256 → 256 → 128 → 3 outputs (classic, e-bike, dock)
```

- 3 hidden layers and about 131,000 parameters in total
- Each output is turned into a chance between 0% and 100%

**How it learned:**
- **Loss:** binary cross-entropy. It rewards the model for giving a high chance when the item was there and a low chance when it wasn't. Only hard cases count.
- **Optimizer:** AdamW
- **Training time:** about 1 minute on a laptop GPU. It stopped by itself after 10 rounds and kept round 7, the best one on validation.

---

## 4. How we score the models: the Brier score

The model gives chances, so we score the chances directly using the **Brier score**:

> Brier = average of (chance we gave − what happened)², where "what happened" is 1 if the item was there and 0 if it wasn't.

- **Lower is better.** 0 is perfect.
- Saying 90% when the bike was there costs very little (0.01). Saying 10% when it was there costs a lot (0.81).
- It rewards a model for being **sure when it should be sure, and unsure when it should be unsure**.

---

## 5. Results (test days, hard cases only)

### Main result

Test Brier score per label (lower is better):

| | App rule | Lookup table | GBM | **MLP** | MLP vs app |
|---|---|---|---|---|---|
| Classic bike | 0.0430 | 0.0395 | 0.0358 | **0.0355** | 17% better |
| E-bike | 0.0404 | 0.0375 | 0.0348 | **0.0345** | 15% better |
| Free dock | 0.0774 | 0.0674 | 0.0614 | **0.0613** | 21% better |

*Source: `train_model.ipynb`, section 9 "Test results", table "Test Brier".*

- **All three models beat the app** on every label.
- **The MLP is the best on every label**, 15–21% better than the app.
- **The MLP beats the GBM only by a hair** (0.0001–0.0003). On this data, the two are about tied.
- The Lookup table, which uses only the count, arrival time and trend, is clearly behind. So the extra inputs (station, time of day, weather) help.

### "Will the bike still be there?"

These are the hard cases where the station shows **1 or 2** bikes right now:

| | Rows | App | Lookup | GBM | **MLP** | MLP vs app |
|---|---|---|---|---|---|---|
| Classic, showing 1–2 | 409,774 | 0.0462 | 0.0413 | 0.0374 | **0.0372** | 19% better |
| showing 1 | 250,542 | 0.0658 | 0.0582 | 0.0525 | **0.0522** | 21% better |
| showing 2 | 159,232 | 0.0153 | 0.0146 | 0.0138 | **0.0137** | 10% better |
| E-bike, showing 1–2 | 524,058 | 0.0615 | 0.0550 | 0.0503 | **0.0499** | 19% better |
| showing 1 | 373,205 | 0.0783 | 0.0695 | 0.0632 | **0.0627** | 20% better |
| showing 2 | 150,853 | 0.0197 | 0.0190 | 0.0182 | **0.0181** | 8% better |

*Computed from `models/test_predictions.parquet`. This table is not in the notebooks.*

- The model helps most when **only 1 bike is showing**, the riskiest case for a rider.
- With 2 showing, the bike is usually still there, so the app is already close.

---

## 6. What this means

1. **The model gives better chances than the app**, 15–21% better by Brier score, and about 20% better when only 1 bike is showing.
2. **The app should show the chance**, or a simple "likely / maybe / unlikely" label, not just yes or no. Otherwise the gain is lost.
3. **The neural network only barely beats the boosted trees.** For the course, we need to show why deep learning is worth it here.

## 7. Ideas for the next experiment

- Give the model a longer history (for example the last 1–2 hours) and try a sequence model such as an **LSTM** or GRU. Right now it only sees 3 points in time, so a sequence model has nothing to work with.
- Add information about nearby stations (for example, a nearby station just emptied).
- Collect more data. We have only about one month, and the test set is only about 4 days.
