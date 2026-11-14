"""One PySR search run (plan Task 4).

Data: train split x search destinations (outer folds 2-4), origins stratified-sampled per region
(N_ORIG per city, seed = SEED). TEMPLATE in {city2, cityses2, shared, single:<REGION>}.
Output: results/sr/L/runs/<scale>_<variant>_<template>_s<seed>/
  config.json, hall_of_fame.csv (complexity, loss, equation); PySR's own checkpoints in pysr/.
Env: SR_NEG_RATE (sampled alternatives for the search; rows of zero-count destinations kept with this
probability, corrected by sample weights 1/rate), SR_TURBO=1 (PySR turbo SIMD evaluation), SR_TIMEOUT
(wall-clock cap in seconds; PySR stops and returns its current front), SR_BLOCK (destination block, default search).
Usage: SR_THREADS=16 python -u run_sr.py SCALE VARIANT TEMPLATE SEED [NITER] [N_ORIG]
"""
import json
import sys

import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pysr_setup import ACTIVE, design_matrix, guesses_for, make_model  # noqa: E402
from sr_data import LINE, ROOT, VARS, build  # noqa: E402


def main(scale, variant, template, seed, niter="100", n_orig="100"):
    """Pooled regions come from SR_REGIONS (default LA,SFBAY = round 1); a city<N> template is renamed to the
    number of pooled regions; single:<city> searches that city alone. Output: results/sr/<LINE>/runs/<tag>."""
    import os
    import re
    seed, niter, n_orig = int(seed), int(niter), int(n_orig)
    pooled = os.environ.get("SR_REGIONS", "LA,SFBAY").split(",")
    regions = [template.split(":")[1]] if template.startswith("single") else pooled
    tpl = "single" if template.startswith("single") else template
    if re.fullmatch(r"city\d+", tpl):
        tpl = f"city{len(regions)}"
    tag = f"{scale}_{variant}_{template.replace(':', '-')}_s{seed}"
    out = ROOT / f"results/sr/{LINE}/runs/{tag}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    neg = os.environ.get("SR_NEG_RATE")
    neg = float(neg) if neg else None
    turbo = os.environ.get("SR_TURBO", "0") == "1"
    dest_block = os.environ.get("SR_BLOCK", "search")      # search = folds 2-4; search23 = folds 2, 3 (plan 4)
    data = build(regions, scale, "train", dest_block, n_orig, seed, offset_B=(variant == "B"), neg_rate=neg,
                 neg_seed=seed)
    procs = int(os.environ.get("SR_THREADS", "16"))
    gpath = os.environ.get("SR_GUESSES")                 # seed expressions (multicity/config/guesses.json)
    guesses = guesses_for(json.loads(Path(gpath).read_text()), variant, tpl) if gpath else None
    extra = {"guesses": guesses} if guesses else {}
    m = make_model(data, variant, tpl, seed, niter, procs, output_directory=str(out), run_id="pysr",
                   temp_equation_file=False, turbo=turbo,
                   timeout_in_seconds=int(os.environ["SR_TIMEOUT"]) if os.environ.get("SR_TIMEOUT") else None, **extra)
    cfg = {"scale": scale, "variant": variant, "template": tpl, "template_arg": template, "regions": regions,
           "line": LINE, "block": dest_block, "seed": seed, "niterations": niter, "n_orig_per_city": n_orig, "rows": int(len(data.y)),
           "origins": int(len(data.offsets) - 1), "vars": VARS, "active_vars": ACTIVE, "threads": procs,
           "neg_rate": neg, "turbo": turbo, "timeout_s": os.environ.get("SR_TIMEOUT"), "guesses": guesses,
           "pysr_params": {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v))
                           for k, v in m.get_params().items() if k not in ("loss_function_expression",)}}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"{tag}: rows {cfg['rows']} origins {cfg['origins']}", flush=True)
    m.fit(design_matrix(data, tpl), data.y, weights=np.exp(data.logw) if neg else None)
    m.equations_[["complexity", "loss", "equation"]].to_csv(out / "hall_of_fame.csv", index=False)
    cfg["seconds"] = round(time.time() - t0)
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    best = m.equations_.sort_values("loss").iloc[0]
    print(f"{tag}: done {cfg['seconds']} s; best loss {best['loss']:.5f} cx {best['complexity']}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
