# 🏁 Pinewood Derby — does our scoring crown the right cars?

An interactive [marimo](https://marimo.io) notebook that Monte-Carlo simulates a
pinewood derby to answer a simple question: **how often does the scoring actually
put the genuinely fastest cars on the podium?**

The schedule is a sliding window of three over randomly-drawn car numbers, so every
car runs each of the three lanes exactly once (lane bias cancels out). The notebook
compares two ways of scoring the event:

- **Placement scoring** (1st/2nd/3rd points only) — how the derby has been scored in past years.
- **Winner-time scoring** — using a new finish-line sensor that captures the winning
  car's time in each heat.

It also explores whether a second pass helps, finds the sweet-spot "championship
round" bracket size, and ends with a race-day **operations guide**.

## ▶️ Run it interactively

- **Live (GitHub Pages):** https://egradman.github.io/pinewood-derby-scoring/
- **Instant preview (molab):**

  [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/egradman/pinewood-derby-scoring/blob/main/notebooks/pinewood_derby.py/wasm)

Both run entirely in your browser via WebAssembly — drag the sliders (cars, heat
noise σ, lane bias) and watch both scoring methods update live.

## 🛠 Run locally

```sh
# with uv (recommended — deps are declared inline in the notebook)
uvx marimo@latest edit notebooks/pinewood_derby.py --sandbox

# or with marimo installed
pip install marimo numpy matplotlib
marimo edit notebooks/pinewood_derby.py
```

## What it found

Capturing the winner's time is the single biggest lever — at realistic noise it
roughly **doubles** the chance of getting the right three cars on the podium and
**triples** the chance of the exact 1-2-3 order, with no extra heats. Re-running the
event with the *same* matchups never helps; a small *targeted* second-round bracket
(re-racing only the top 4–6 cars) is the best use of one extra pass when no sensor is
available.
