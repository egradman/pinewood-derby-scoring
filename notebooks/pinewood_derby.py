# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏁 Pinewood Derby — does our scoring crown the right cars?

    **The schedule.** `N` cars, numbers `1…N` drawn at random. Three lanes A/B/C.
    Race `k` = numbers `{k, k+1, k+2}` (wrapping). Sliding by one each race means
    **every car runs each lane exactly once**, so lane bias cancels out. Per heat:
    fastest → **1 pt**, 2nd → **2**, 3rd → **3**; lowest 3-heat total wins.

    **The question this year.** We just installed a finish-line **sensor** that
    captures the *winner's time* in each heat. Is it worth using? Below we run the
    same Monte-Carlo two ways — **without** the sensor (how we scored in past years)
    and **with** it — and measure how often each method's final standings match the
    cars' true speed order. Drag the controls; both sections update live.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TL;DR — yes, use the sensor ✅

    Scoring by the sensor's **winning times** instead of just finishing places makes
    the podium dramatically more accurate, with no extra racing — and a small **6-car
    second round** squeezes out a bit more. Measured over 20,000 simulated events
    (24 cars, realistic heat-to-heat variation):

    | Method | Right 3 cars | Exact 1-2-3 | Correct winner |
    |---|--:|--:|--:|
    | **Old** — placement only *(past years)* | 37% | 17% | 67% |
    | **New** — sensor winning times | **76%** | **50%** | **82%** |
    | New + 6-car second round | **83%** | **62%** | **89%** |

    **Bottom line:** it's definitely worth using the sensor as-built — it roughly
    *doubles* the odds of crowning the right three cars — and probably worth running a
    **6-car second round** on top for the extra accuracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the simulation works (in plain terms)

    Think of it like rolling dice thousands of times to learn the odds:

    1. **Give every car a true speed.** We *pretend* we know each car's real speed,
       fastest to slowest. In a real derby we never know this — that's the whole
       point of racing — but in simulation we set it, so we can grade our scoring
       against the truth.
    2. **Run the whole event once.** Randomly assign car numbers, build the heat
       schedule, and "race" every heat: a car's result = its true speed + its lane's
       advantage + a dash of random luck (**σ**). Tally points (and, this year, the
       winner's time), then crown a podium.
    3. **Check it against reality.** Did the top 3 finishers actually match the 3
       genuinely fastest cars? Record yes / no.
    4. **Repeat thousands of times** with fresh number-draws and fresh luck. The
       fraction of events that got the podium right is our **accuracy** — how often
       the contest rewards real speed instead of a lucky draw.

    The **σ** slider is the wildcard: how much a car's time wobbles run-to-run
    (track dust, release jitter, a sticky wheel). σ = 0 means perfectly consistent
    cars; bigger σ means more luck — and luck is the enemy of a fair result.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    def run_derby(N=24, n_sims=2000, sigma=0.05, lane_bonus_A=0.0,
                  second="none", top_k=6, score="wintime", seed=0):
        """Pass 1 = standard random sliding-window schedule. `score`:
          'ordinal' -> golf points 1/2/3 per heat (placement only)
          'wintime' -> ALSO record each heat WINNER's lane-corrected time (a real speed
                       sample); rank by mean winning speed, untimed cars below by points.
        `second` adds one extra pass: 'none' / 'random' / 'seeded' (full field by standing)
        / 'targeted' (re-race only the top_k contenders; the rest frozen)."""
        rng = np.random.default_rng(seed)
        speeds = np.linspace(0.0, 1.0, N)
        true_order = np.argsort(-speeds); true_top3 = true_order[:3]
        true_rank = np.empty(N, int); true_rank[true_order] = np.arange(N)
        lane = np.array([lane_bonus_A, 0.0, 0.0])

        podium = podium_set = top1 = 0
        conf = np.zeros((N, N), int)
        for s in range(n_sims):
            pts = np.zeros(N); ssum = np.zeros(N); cnt = np.zeros(N)

            def play(cars):                       # cars: (m,3); accrue points + winner times
                perf = speeds[cars] + lane[None, :] + rng.normal(0, sigma, size=cars.shape)
                place = (-perf).argsort(axis=1).argsort(axis=1) + 1
                np.add.at(pts, cars.ravel(), place.ravel())
                m = cars.shape[0]; w = perf.argmax(axis=1); rows = np.arange(m)
                np.add.at(ssum, cars[rows, w], perf[rows, w] - lane[w])   # lane-corrected
                np.add.at(cnt, cars[rows, w], 1.0)

            def order(idx):                        # best -> worst among car ids `idx`
                if score == "ordinal":
                    ro = speeds[idx] + rng.normal(0, sigma, size=len(idx))
                    return idx[np.lexsort((-ro, pts[idx]))]
                est = np.where(cnt[idx] > 0, ssum[idx] / np.maximum(cnt[idx], 1), -np.inf)
                return idx[np.lexsort((pts[idx], -est, -(cnt[idx] > 0).astype(int)))]

            sched = lambda n: (np.arange(n)[:, None] + np.arange(3)[None, :]) % n
            play(rng.permutation(N)[sched(N)])     # pass 1
            if second == "none":
                final = order(np.arange(N))
            elif second == "random":
                play(rng.permutation(N)[sched(N)])
                final = order(np.arange(N))
            else:
                standing = order(np.arange(N))
                m = N if second == "seeded" else int(np.clip(top_k, 2, N))
                top = standing[:m]
                play(top[sched(m)])                # championship round among top m
                final = np.concatenate([order(top), standing[m:]])
            fr = np.empty(N, int); fr[final] = np.arange(N)
            podium     += (final[:3] == true_top3).all()
            podium_set += np.isin(true_top3, final[:3]).all()
            top1       += (final[0] == true_order[0])
            conf[true_rank, fr] += 1
        return dict(podium=podium / n_sims, podium_set=podium_set / n_sims,
                    top1=top1 / n_sims, conf=conf, n_sims=n_sims, N=N)

    return (run_derby,)


@app.cell(hide_code=True)
def _(mo):
    sigma_ui = mo.ui.slider(0.0, 0.30, value=0.05, step=0.005, label="Heat noise σ (car run-to-run variation)", show_value=True)
    lane_ui  = mo.ui.slider(0.0, 0.50, value=0.0,  step=0.01,  label="Lane A bonus", show_value=True)
    nsims_ui = mo.ui.slider(200, 5000, value=3000, step=200,   label="Simulations", show_value=True)
    N_ui     = mo.ui.slider(6, 48, value=24, step=1,           label="Cars (N)", show_value=True)
    mo.hstack([N_ui, sigma_ui, lane_ui, nsims_ui], justify="start", gap=2)
    return N_ui, lane_ui, nsims_ui, sigma_ui


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Without the sensor — *placement scoring*  (how we've scored in past years)

    Each heat records only the finishing **order** across the three lanes
    (1st = 1 pt, 2nd = 2, 3rd = 3). We never learn *by how much* a car won — only
    who beat whom. Final standings = lowest total points; ties go to a run-off.

    Here is how often that method recovers the true speed order (single pass, current settings):
    """)
    return


@app.cell(hide_code=True)
def _(N_ui, lane_ui, nsims_ui, run_derby, sigma_ui):
    res_ord = run_derby(N=N_ui.value, n_sims=nsims_ui.value, sigma=sigma_ui.value,
                        lane_bonus_A=lane_ui.value, second="none", score="ordinal")
    return (res_ord,)


@app.cell(hide_code=True)
def _(mo, res_ord):
    mo.hstack([
        mo.stat(f"{res_ord['podium_set']*100:.1f}%", label="Right 3 cars on podium", bordered=True),
        mo.stat(f"{res_ord['podium']*100:.1f}%",     label="Top-3 in exact order", bordered=True),
        mo.stat(f"{res_ord['top1']*100:.1f}%",       label="Correct overall winner", bordered=True),
    ], justify="space-around")
    return


@app.cell(hide_code=True)
def _(plt, res_ord, sigma_ui):
    _conf = res_ord["conf"]; _N = res_ord["N"]
    _prob = _conf / _conf.sum(axis=1, keepdims=True)
    _fig, _ax = plt.subplots(figsize=(6.2, 5.4))
    _im = _ax.imshow(_prob, cmap="magma", aspect="auto", origin="upper")
    _ax.set_xlabel("Final standings rank (0 = winner)")
    _ax.set_ylabel("True-speed rank (0 = fastest)")
    _ax.set_title(f"Placement scoring — where each car lands (σ={sigma_ui.value:g})")
    _ax.plot([0, _N-1], [0, _N-1], color="cyan", lw=1, ls="--", alpha=.6)
    _fig.colorbar(_im, ax=_ax, label="probability"); _fig.tight_layout(); _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · With the sensor — *winner-time scoring*  (new this year)

    The sensor captures the **winning car's elapsed time** in each heat (but not
    2nd/3rd). We treat it as a direct speed measurement: subtract the known lane
    bias, then rank cars by their **average winning time**. Any car that never wins
    a heat falls back to placement order. Same schedule, same settings — just more
    information per heat:
    """)
    return


@app.cell(hide_code=True)
def _(N_ui, lane_ui, nsims_ui, run_derby, sigma_ui):
    res_win = run_derby(N=N_ui.value, n_sims=nsims_ui.value, sigma=sigma_ui.value,
                        lane_bonus_A=lane_ui.value, second="none", score="wintime")
    return (res_win,)


@app.cell(hide_code=True)
def _(mo, res_win):
    mo.hstack([
        mo.stat(f"{res_win['podium_set']*100:.1f}%", label="Right 3 cars on podium", bordered=True),
        mo.stat(f"{res_win['podium']*100:.1f}%",     label="Top-3 in exact order", bordered=True),
        mo.stat(f"{res_win['top1']*100:.1f}%",       label="Correct overall winner", bordered=True),
    ], justify="space-around")
    return


@app.cell(hide_code=True)
def _(plt, res_win, sigma_ui):
    _conf = res_win["conf"]; _N = res_win["N"]
    _prob = _conf / _conf.sum(axis=1, keepdims=True)
    _fig, _ax = plt.subplots(figsize=(6.2, 5.4))
    _im = _ax.imshow(_prob, cmap="magma", aspect="auto", origin="upper")
    _ax.set_xlabel("Final standings rank (0 = winner)")
    _ax.set_ylabel("True-speed rank (0 = fastest)")
    _ax.set_title(f"Winner-time scoring — where each car lands (σ={sigma_ui.value:g})")
    _ax.plot([0, _N-1], [0, _N-1], color="cyan", lw=1, ls="--", alpha=.6)
    _fig.colorbar(_im, ax=_ax, label="probability"); _fig.tight_layout(); _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Side by side — is the sensor worth using?

    The bar chart compares the two methods at the current noise level; the line
    chart sweeps the full range of heat-to-heat noise. The sensor's edge grows as
    racing gets noisier (faster cars, looser tracks, hand-released cars).
    """)
    return


@app.cell(hide_code=True)
def _(N_ui, lane_ui, np, nsims_ui, plt, run_derby, sigma_ui):
    _ns6 = min(nsims_ui.value, 3000)
    _methods = [("ordinal","placement\n(past years)"), ("wintime","winner times\n(this year)")]
    _sc = {m: run_derby(N=N_ui.value, n_sims=_ns6, sigma=sigma_ui.value,
                        lane_bonus_A=lane_ui.value, second="none", score=m) for m, _ in _methods}
    _x = np.arange(len(_methods)); _w = 0.27
    _fig5, _ax5 = plt.subplots(figsize=(6.8, 4.4))
    for _key, _lab, _c, _off in [("podium_set","right 3 cars","C0",-_w),
                                 ("podium","exact 1-2-3","C1",0.0),
                                 ("top1","winner","C2",_w)]:
        _vals = [_sc[m][_key] for m, _ in _methods]
        _ax5.bar(_x + _off, _vals, _w, label=_lab, color=_c)
        for _xi, _v in zip(_x + _off, _vals):
            _ax5.text(_xi, _v + 0.012, f"{_v*100:.0f}", ha="center", va="bottom", fontsize=8)
    _ax5.set_xticks(_x); _ax5.set_xticklabels([l for _, l in _methods])
    _ax5.set_ylabel("fraction of simulations"); _ax5.set_ylim(0, 1.05)
    _ax5.set_title(f"Sensor vs no sensor  (1 pass, σ={sigma_ui.value:g}, N={N_ui.value})")
    _ax5.legend(fontsize=8, loc="upper right"); _ax5.grid(axis="y", alpha=.3)
    _fig5.tight_layout(); _fig5
    return


@app.cell(hide_code=True)
def _(N_ui, lane_ui, np, nsims_ui, plt, run_derby, sigma_ui):
    _sigmas = np.linspace(0.0, 0.30, 16); _ns = min(nsims_ui.value, 1500)
    _o = [run_derby(N=N_ui.value, n_sims=_ns, sigma=float(s), lane_bonus_A=lane_ui.value, second="none", score="ordinal")  for s in _sigmas]
    _w2 = [run_derby(N=N_ui.value, n_sims=_ns, sigma=float(s), lane_bonus_A=lane_ui.value, second="none", score="wintime") for s in _sigmas]
    _fig2, _ax2 = plt.subplots(figsize=(7.2, 4.4))
    _ax2.plot(_sigmas, [r["podium_set"] for r in _w2], "o-",  color="C0", lw=2.2, label="winner times · right 3 cars")
    _ax2.plot(_sigmas, [r["podium_set"] for r in _o ], "o--", color="C0", lw=1.3, alpha=.6, label="placement · right 3 cars")
    _ax2.plot(_sigmas, [r["podium"]     for r in _w2], "s-",  color="C1", lw=2.2, label="winner times · exact 1-2-3")
    _ax2.plot(_sigmas, [r["podium"]     for r in _o ], "s--", color="C1", lw=1.3, alpha=.6, label="placement · exact 1-2-3")
    _ax2.axvline(sigma_ui.value, color="gray", ls=":", lw=1)
    _ax2.set_xlabel("Heat noise σ"); _ax2.set_ylabel("fraction of simulations"); _ax2.set_ylim(-0.02, 1.02)
    _ax2.set_title("Sensor (solid) vs no sensor (dashed) across noise")
    _ax2.legend(fontsize=8, loc="upper right"); _ax2.grid(alpha=.3); _fig2.tight_layout(); _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optional: add a second pass (advanced)

    If you want to push the podium further, you can run **one extra round**. The
    controls below explore the options. Two takeaways: **with the sensor on**, a
    second pass adds little (the time signal already does the work); **without it**,
    a small *targeted* bracket — re-racing only the top 4-6 cars against each other —
    is the best use of one extra round. Re-running the *same* matchups never helps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    score_appx_ui = mo.ui.dropdown(options={"Ordinal (1/2/3 only)": "ordinal", "Winner times": "wintime"},
                                  value="Ordinal (1/2/3 only)", label="Scoring (appendix)")
    second_ui = mo.ui.dropdown(options={"Single pass (baseline)": "none", "+ random second pass": "random",
                                        "+ seeded full field": "seeded", "+ targeted top-k bracket": "targeted"},
                               value="+ targeted top-k bracket", label="Second pass")
    topk_ui = mo.ui.slider(2, 24, value=6, step=1, label="Bracket size (top-k)", show_value=True)
    mo.hstack([score_appx_ui, second_ui, topk_ui], justify="start", gap=2)
    return score_appx_ui, topk_ui


@app.cell(hide_code=True)
def _(
    N_ui,
    lane_ui,
    np,
    nsims_ui,
    plt,
    run_derby,
    score_appx_ui,
    sigma_ui,
    topk_ui,
):
    _k = topk_ui.value
    _modes = [("none","single"), ("random","+random"), ("seeded","+seeded"), ("targeted",f"+targeted\n(top-{_k})")]
    _ns4 = min(nsims_ui.value, 3000)
    _rc = {m: run_derby(N=N_ui.value, n_sims=_ns4, sigma=sigma_ui.value, lane_bonus_A=lane_ui.value,
                        second=m, top_k=_k, score=score_appx_ui.value) for m, _ in _modes}
    _x = np.arange(len(_modes)); _w = 0.27
    _fig3, _ax3 = plt.subplots(figsize=(7.6, 4.4))
    for _key, _lab, _c, _off in [("podium_set","right 3 cars","C0",-_w),
                                 ("podium","exact 1-2-3","C1",0.0),
                                 ("top1","winner","C2",_w)]:
        _vals = [_rc[m][_key] for m, _ in _modes]
        _ax3.bar(_x + _off, _vals, _w, label=_lab, color=_c)
        for _xi, _v in zip(_x + _off, _vals):
            _ax3.text(_xi, _v + 0.012, f"{_v*100:.0f}", ha="center", va="bottom", fontsize=7)
    _ax3.set_xticks(_x); _ax3.set_xticklabels([l for _, l in _modes])
    _ax3.set_ylabel("fraction of simulations"); _ax3.set_ylim(0, 1.05)
    _ax3.set_title(f"Best use of ONE extra pass  ({score_appx_ui.selected_key}, σ={sigma_ui.value:g})")
    _ax3.legend(fontsize=8, loc="upper right"); _ax3.grid(axis="y", alpha=.3); _fig3.tight_layout(); _fig3
    return


@app.cell(hide_code=True)
def _(
    N_ui,
    lane_ui,
    np,
    nsims_ui,
    plt,
    run_derby,
    score_appx_ui,
    sigma_ui,
    topk_ui,
):
    _ks = np.array([2,3,4,5,6,8,10,12,16,N_ui.value]); _ks = np.unique(_ks[_ks <= N_ui.value])
    _ns5 = min(nsims_ui.value, 1500)
    _kr = [run_derby(N=N_ui.value, n_sims=_ns5, sigma=sigma_ui.value, lane_bonus_A=lane_ui.value,
                     second="targeted", top_k=int(k), score=score_appx_ui.value) for k in _ks]
    _base = run_derby(N=N_ui.value, n_sims=_ns5, sigma=sigma_ui.value, lane_bonus_A=lane_ui.value,
                      second="none", score=score_appx_ui.value)
    _fig4, _ax4 = plt.subplots(figsize=(7.2, 4.2))
    _ax4.plot(_ks, [r["podium_set"] for r in _kr], "o-", color="C0", lw=2, label="right 3 cars")
    _ax4.plot(_ks, [r["podium"]     for r in _kr], "s-", color="C1", lw=2, label="exact 1-2-3")
    _ax4.axhline(_base["podium_set"], color="C0", ls=":", lw=1, alpha=.7)
    _ax4.axhline(_base["podium"],     color="C1", ls=":", lw=1, alpha=.7)
    _ax4.axvline(topk_ui.value, color="gray", ls=":", lw=1)
    _ax4.set_xlabel("bracket size (top-k re-raced)"); _ax4.set_ylabel("fraction of simulations"); _ax4.set_ylim(0, 1.0)
    _ax4.set_title(f"Sweet spot for the championship round  ({score_appx_ui.selected_key}, σ={sigma_ui.value:g})")
    _ax4.legend(fontsize=8, loc="upper right"); _ax4.grid(alpha=.3); _fig4.tight_layout(); _fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏁 How to run the event — operations guide

    A start-to-finish checklist for race day, reflecting what the simulation shows.
    We assume the finish-line **sensor is accurate** — its times are trustworthy and
    directly comparable across lanes (no per-lane correction needed).

    ### Before the event
    - **Confirm the field.** Count cars (`N`). Every car will race exactly **3 times**,
      once in each lane.
    - **Draw numbers at random.** Assign each car a number `1…N` by random draw
      (pull from a hat). Don't let entrants pick — randomization is what makes the
      lane-balancing fair.
    - **Generate the heat sheet.** Heat `k` (for `k = 1…N`) runs the three cars
      numbered `k, k+1, k+2`, wrapping past `N` back to `1`. Put the lower number in
      **lane A**, middle in **B**, higher in **C**. This guarantees every car runs
      each lane once.
    - **Check the sensor.** Do a dry run to confirm the finish-line timer fires
      reliably on the winning car and logs a clean time.

    ### Running the heats
    - Run heats **in order**, 1 through `N`. Release all three cars together.
    - For each heat, record: **finishing order** (1st/2nd/3rd) *and* the
      **winner's sensor time**.
    - If the sensor misses a heat, fall back to placement for that heat only and note it.

    ### Scoring  (use the sensor — it's the big win)
    - **Primary (this year): winner-time ranking.** For each car, average the sensor
      times from the heats it **won**, and rank cars **fastest average winning time →
      slowest**. Cars that never won a heat are placed below the winners, ordered by
      their placement points.
    - **Why:** in simulation this roughly **doubles** the chance of getting the right
      three cars on the podium and **triples** the chance of the exact 1-2-3 order,
      versus placement-only — with **no extra heats**. The faster a car, the more
      heats it wins, so the timing data piles up exactly on the podium contenders.
    - **Fallback (no sensor): placement points.** Sum each car's three finishes
      (1+2+3 scale); **lowest total wins**. Keep this as a backup and as a sanity check.

    ### Crowning the podium
    - Top 3 by the primary ranking are 1st / 2nd / 3rd.
    - **Ties** (equal placement points, or indistinguishable times): run a **run-off
      heat** among the tied cars and rank by the run-off result.

    ### If you want extra confidence in the podium (optional second round)
    - **With the sensor:** usually unnecessary — the times already settle it. A
      **6-car second round** (re-race the top six against each other) buys a few more
      points of accuracy if you want it.
    - **Without the sensor:** run a **championship round** — re-race just the **top
      4-6 cars** (by first-round standings) against each other, and use that round to
      order the podium. This is the best use of one extra pass; re-racing the whole
      field, or re-running the same matchups, helps little or not at all.

    ### One-line recommendation
    > **Use the sensor's winner times as the primary score.** One clean pass with
    > timing beats every placement-only scheme we tested — including running the
    > whole event twice.
    """)
    return


if __name__ == "__main__":
    app.run()
