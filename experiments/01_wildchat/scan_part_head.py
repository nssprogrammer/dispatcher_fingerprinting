# %% [markdown]
# # Notebook 4 — Closing the WildChat analysis: a server-cap test, and a scan with exact p-values
# Notebook 3 left three problems. This notebook fixes them and closes the WildChat thread.
#
# **A. Is there a site-wide concurrency cap?** WildChat gives only response end times, so intervals are *generation only*. If WildChat's backend processed only a few requests at once, the whole site would take turns, and pools that contain most of the site (1-hour pools) would read R < 1. Tests: the site-wide Fano factor and the share of time with 2+ requests in flight, against the Poisson expectation, by load decile; plus the same with durations halved.
#
# **B. A scan whose error rate we can trust.**
# - Exact one-sided surrogate p-values (100 time-shifts, escalated to 1,100 when p ≤ 0.05) instead of z-scores.
# - *Diverse* controls: activity-matched accounts drawn from different user-agent+country cells and different persons, so a control cannot contain one operator's IPs.
# - Benjamini–Hochberg false-discovery control.
# - A corrected oracle: a prompt prefix is 'generic' only if it is used in ≥ 5 different user-agent+country cells (in Notebook 3 an operator with > 20 IPs made its own template 'generic').
#
# **Pre-registered reading.** Controls should give p-values close to uniform (≈1% below 0.01). The scan supports real hidden operators only if real groups yield BH discoveries while controls yield about none, and the oracle rate among discoveries exceeds the rate among non-discoveries.
#
# **Run settings.** Accelerator *None*, Internet *On*. Uses all CPU cores. Expected runtime 25–45 min.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, glob
T_START = time.time()
CFG = dict(repo="allenai/WildChat-1M", n_shards=14, data_dir=os.environ.get("NB4_DATA_DIR", ""),
           tau=60.0, max_dur_s=900.0, max_component_ips=50,
           generic_cells=5,                 # a prefix used in >= this many user-agent+country cells is generic
           scan_min_accounts=5, scan_min_requests=30,
           n_sur_initial=100, n_sur_full=1000, escalate_p=0.05, fdr_q=0.05,
           n_jobs=os.cpu_count(), seed=11)
print(json.dumps(CFG, indent=1))

