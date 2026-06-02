import os
import numpy as np
import pandas as pd
import arviz as az
from cmdstanpy import CmdStanModel
import cmdstanpy as csp
from plotnine import (
    ggplot, aes, geom_line, facet_wrap, scale_x_log10,
    labs, theme_bw, theme, element_text, scale_y_log10
)
csp.disable_logging()

NU_TARGET = 5
NU_INIT = 2
N_DRAWS = 256
N_REPLICATIONS = 128
M_VALUES = [4, 16, 64]
M_MAX = max(M_VALUES)
ITER_STRIDE = 1
ITER_RANGE = range(8, N_DRAWS + 1, ITER_STRIDE)
STEP_SIZE = 0.8
INIT_CLIP = 50.0
RHAT_METHOD = "rank"
BASE_SEED = 20260601

TARGET_M1 = 0.0
TARGET_M2 = NU_TARGET / (NU_TARGET - 2.0)   # E[theta^2] = variance = 5/3
TARGET_M3 = 0.0                             

STAN_CODE = """
data { real<lower=0> nu; }
parameters { real theta; }
model { theta ~ student_t(nu, 0, 1); }
"""

# (label, row, column, target, is_stein)
# (label, row, column, target, is_stein)
FACET_SPEC = [
    ("RMSE E[theta]",        "error", "theta",   0.0, False),
    ("RMSE E[theta^2]",      "error", "theta^2", 0.0, False),
    ("RMSE E[theta^3]",      "error", "theta^3", 0.0, False),
    ("ME R-hat - 1 (theta)",         "rhat",  "theta",   1.0, False),
    ("ME R-hat - 1 (theta^2)",       "rhat",  "theta^2", 1.0, False),
    ("ME R-hat - 1 (theta^3)",       "rhat",  "theta^3", 1.0, False),
    ("RMSE Stein(k=0) (theta)",    "stein", "theta",   0.0, True),
    ("RMSE Stein(k=1) (theta^2)",  "stein", "theta^2", 0.0, True),
    ("RMSE Stein(k=2) (theta^3)",  "stein", "theta^3", 0.0, True),
]
FACET_ORDER = [s[0] for s in FACET_SPEC]
STEIN_FACETS = {s[0] for s in FACET_SPEC if s[4]}
LOG_2_X_BREAKS = np.arange(3, 12)
X_BREAKS = 2**LOG_2_X_BREAKS
X_LABELS = [rf"$2^{{{b}}}$" for b in LOG_2_X_BREAKS]


def build_model(path="student_t5.stan"):
    with open(path, "w") as f:
        f.write(STAN_CODE)
    return CmdStanModel(stan_file=path)


def heavy_tailed_inits(n_chains, rng):
    draws = rng.standard_t(NU_INIT, size=n_chains)
    while np.any(np.abs(draws) > INIT_CLIP):
        bad = np.abs(draws) > INIT_CLIP
        draws[bad] = rng.standard_t(NU_INIT, size=int(bad.sum()))
    return draws


def run_replication(model, rng, seed):
    inits = heavy_tailed_inits(M_MAX, rng)
    fit = model.sample(
        data={"nu": NU_TARGET},
        chains=M_MAX,
        parallel_chains=os.cpu_count(),
        inits=[{"theta": float(v)} for v in inits],
        iter_warmup=0,
        iter_sampling=N_DRAWS,
        adapt_engaged=False,
        step_size=STEP_SIZE,
        seed=seed,
        show_progress=False,
        show_console=False,
    )
    draws = fit.draws(concat_chains=False)
    theta_col = fit.column_names.index("theta")
    return draws[:, :, theta_col]          # (N_DRAWS, M_MAX)


def student_t_score(theta, nu=NU_TARGET):
    return -(nu + 1.0) * theta / (nu + theta**2)


def rhat(values_by_chain):
    return float(az.rhat(values_by_chain, method=RHAT_METHOD))


def stein_error(accumuland):
    return accumuland.mean(axis=0).mean()

def quantities_for_slice(theta_slice):     # (n, M)
    by_chain = theta_slice.T
    s = student_t_score(theta_slice)
    pooled = theta_slice.reshape(-1)
    return {
        "RMSE E[theta]":       (pooled.mean() - TARGET_M1) ** 2,
        "RMSE E[theta^2]":     ((pooled**2).mean() - TARGET_M2) ** 2,
        "RMSE E[theta^3]":     ((pooled**3).mean() - TARGET_M3) ** 2,
        "ME R-hat - 1 (theta)":        rhat(by_chain),
        "ME R-hat - 1 (theta^2)":      rhat(by_chain**2),
        "ME R-hat - 1 (theta^3)":      rhat(by_chain**3),
        "RMSE Stein(k=0) (theta)":   stein_error(s) ** 2,
        "RMSE Stein(k=1) (theta^2)": stein_error(1.0 + theta_slice * s) ** 2,
        "RMSE Stein(k=2) (theta^3)": stein_error(2.0 * theta_slice + theta_slice**2 * s) ** 2,
    }

def simulate():
    model = build_model()
    rng = np.random.default_rng(BASE_SEED)
    records = []
    for rep in range(N_REPLICATIONS):
        print(f"Replication {(rep + 1):6d} / {N_REPLICATIONS:6d}")
        theta = run_replication(model, rng, seed=BASE_SEED + rep)
        for M in M_VALUES:
            for n in ITER_RANGE:
                sl = theta[:n, :M]
                for name, value in quantities_for_slice(sl).items():
                    records.append((rep, M, n, name, value))
    return pd.DataFrame(records, columns=["rep", "M", "n", "quantity", "value"])

def aggregate(raw):
    agg = (raw.groupby(["quantity", "M", "n"], observed=True)["value"]
              .mean()
              .reset_index())
    agg["quantity"] = pd.Categorical(agg["quantity"],
                                     categories=FACET_ORDER, ordered=True)
    agg["M"] = pd.Categorical(agg["M"], categories=M_VALUES, ordered=True)
    return agg


def target_frame():
    rows = [(label, target) for (label, _r, _c, target, _s) in FACET_SPEC]
    df = pd.DataFrame(rows, columns=["quantity", "target"])
    df["quantity"] = pd.Categorical(df["quantity"],
                                    categories=FACET_ORDER, ordered=True)
    return df

def plot_results(agg, outfile="stein_grid.png"):
    label_to_row = {label: row for (label, row, _c, _t, _s) in FACET_SPEC}
    df = agg.copy()
    df["row"] = df["quantity"].map(label_to_row)

    is_error = df["row"] == "error"
    is_rhat = df["row"] == "rhat"
    is_stein = df["row"] == "stein"
    df.loc[is_error, "value"] = np.sqrt(df.loc[is_error, "value"])   # MSE -> RMSE
    df.loc[is_stein, "value"] = np.sqrt(df.loc[is_stein, "value"])   # MS bias -> RMS bias
    df.loc[is_rhat, "value"] = df.loc[is_rhat, "value"] - 1.0        # R-hat -> R-hat - 1

    df = df[df["value"] > 0]

    p = (ggplot(df, aes("n", "value", color="M"))
         + geom_line()
         + facet_wrap("~quantity", ncol=3, scales="free_y")
         + scale_x_log10(
               breaks=X_BREAKS,
               labels=lambda bs: [rf"$2^{{{int(round(np.log2(b)))}}}$" for b in bs])
         + scale_y_log10()
         + labs(x="iterations per chain",
                y="distance from target",
                color="chains M",
                title=f"t({NU_TARGET}) target, t({NU_INIT}) init")
         + theme_bw()
         + theme(figure_size=(11, 8),
                 plot_title=element_text(size=11),
                 strip_text=element_text(size=8)))
    p.save(outfile, dpi=150, verbose=False)
    print(f"saved plot to {outfile}")
    return p

if __name__ == "__main__":
    raw = simulate()
    agg = aggregate(raw)
    plot_results(agg)
