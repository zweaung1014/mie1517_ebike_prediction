# Experiment 1: Summary

A short update on the first full run: what we built, how we tested it, and what we found.

Notebooks: `build_dataset.ipynb` → `train_model.ipynb` → `analyze_accuracy.ipynb`. Saved model and results are in `models/`.

---

## 1. What we are trying to do

A rider opens the Bike Share Toronto app and sees "1 e-bike" at a station 10 minutes away. Will that e-bike still be there when they arrive?

The app only shows what is there **right now**. It doesn't predict anything. Our model tries to answer:

> "If I get to station S in **h** minutes (2 to 15), will there be at least one **classic bike**, at least one **e-bike**, and at least one **free dock**?"

The model gives a **chance** for each of the three (for example "e-bike: 72%"), not just yes or no.

We only use the model for the **hard cases**: when the station shows 0, 1 or 2 of something. If a station shows 3 or more, we just say "yes", because it is almost always still there a few minutes later.

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

## 3. The model

A simple neural network called an **MLP** (multi-layer perceptron). One network is shared by all stations.

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
- **Loss:** binary cross-entropy. It rewards the model for giving a high chance when the bike was there and a low chance when it wasn't. Only the hard cases count.
- **Optimizer:** AdamW
- **Training time:** about 1 minute on a laptop GPU. It stopped by itself after 10 rounds and kept round 7, the best one on validation.

**What we compared it against:**
1. **App rule:** "yes if the count right now is at least 1". This is what riders do today.
2. **Lookup table:** the past success rate for the same count, arrival time and recent trend.
3. **Gradient-boosted trees:** a strong standard model for table data.

---

## 4. Results (test days, hard cases only)

### 4a. How good are the chances?

We use the **Brier score**: how far, on average, the chance we gave was from what really happened. **Lower is better.**

| | App rule | Lookup table | Boosted trees | **Our MLP** | MLP vs app |
|---|---|---|---|---|---|
| Classic bike | 0.0430 | 0.0395 | 0.0358 | **0.0355** | 17% better |
| E-bike | 0.0404 | 0.0375 | 0.0348 | **0.0345** | 15% better |
| Free dock | 0.0774 | 0.0674 | 0.0614 | **0.0613** | 21% better |

- The MLP is the best of all four on every label.
- It beats the app by 15–21%.
- It beats the boosted trees only **by a hair**. On this data, the two are about equally good.

### 4b. How often is a plain "yes / no" right?

Here the model says "yes" when its chance is above a cutoff, and we count right and wrong answers.

| | Examples | App correct | MLP correct (best cutoff) | MLP correct (70% cutoff) |
|---|---|---|---|---|
| Classic bike | 894,074 | 95.70% | 95.70% | 95.37% |
| E-bike | 1,609,180 | 95.96% | 95.97% | 95.67% |
| Free dock | 252,537 | 92.26% | 92.23% | 91.84% |

- **As a yes/no answer, the model ties with the app.** The reason is that the count right now already decides most answers. If a station shows 0 classic bikes, 96% of the time there are still none later. If it shows 1, it is still there 93% of the time.
- **A stricter 70% cutoff** cuts wrong "yes" answers (wasted trips) by about 12–23%. But it adds more wrong "no" answers, so overall it gets a few thousand fewer right per label.

---

## 5. What this means

1. **The model's value is in the chance, not the yes/no.** It knows *how sure* to be. For example, "1 bike now, arriving in 14 minutes" should get a lower chance than "2 bikes now, arriving in 3 minutes." A plain yes/no throws that away.
2. **So the app should show the chance**, or a simple "likely / maybe / unlikely" label, not just yes or no.
3. **The neural network only barely beats the boosted trees.** For the course, we need to show why deep learning is worth it here.

## 6. Ideas for the next experiment

- Give the model a longer history (for example the last 1–2 hours) and try a sequence model such as an **LSTM** or GRU. Right now it only sees 3 points in time, so a sequence model has nothing to work with.
- Add information about nearby stations (for example, a nearby station just emptied).
- Collect more data. We have only about one month, and the test set is only about 4 days.
- Try a "likely / maybe / unlikely" answer and check how often each one is right.
