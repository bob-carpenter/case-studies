import numpy as np
import pandas as pd
import arviz as az
from plotnine import (
    ggplot, aes, geom_line, facet_wrap, scale_x_log10,
    labs, theme_bw, theme, element_text, scale_y_log10
)

N_DRAWS = 256
N_REPLICATIONS = 128
M_VALUES = [4, 16, 64]
M_MAX = max(M_VALUES)
ITER_STRIDE = 1
ITER_RANGE = range(8, N_DRAWS + 1, ITER_STRIDE)
RHAT_METHOD = "rank"
BASE_SEED = 20260601

# Cauchy has no finite moments; nominal targets are principal-value locations
TARGET_M1 = 0.0
TARGET_M2 = 0.0   # E[theta^2] = infinity; RMSE will show chaos
TARGET_M3 = 0.0

TANH_SCALES = [0.5, 1.0, 2.0]   # c in tanh(theta/c)

FACET_SPEC = [
    ("RMSE E[theta]",              "error", "col1", 0.0, False),
    ("RMSE E[theta^2]",            "error", "col2", 0.0, False),
    ("RMSE E[theta^3]",            "error", "col3", 0.0, False),
    ("ME R-hat - 1 (theta)",       "rhat",  "col1", 1.0, False),
    ("ME R-hat - 1 (theta^2)",     "rhat",  "col2", 1.0, False),
    ("ME R-hat - 1 (theta^3)",     "rhat",  "col3", 1.0, False),
    ("RMSE Stein tanh(theta/0.5)", "stein", "col1", 0.0, True),
    ("RMSE Stein tanh(theta)",     "stein", "col2", 0.0, True),
    ("RMSE Stein tanh(theta/2)",   "stein", "col3", 0.0, True),
]
FACET_ORDER = [s[0] for s in FACET_SPEC]
LOG_2_X_BREAKS = np.arange(3, 12)
X_BREAKS = 2**LOG_2_X_BREAKS


def cauchy_score(theta):
    return -2.0 * theta / (1.0 + theta**2)


def tanh_stein_accumuland(theta, c):
    s = cauchy_score(theta)
    sech2 = 1.0 / np.cosh(theta / c)**2
    return sech2 / c + np.tanh(theta / c) * s


def rhat_safe(values_by_chain):
    # clip before ranking to avoid inf/NaN from extreme Cauchy draws
    clipped = np.clip(values_by_chain, -1e10, 1e10)
    return float(az.rhat(clipped, method=RHAT_METHOD))


def stein_error(accumuland):
    return accumuland.mean(axis=0).mean()


def run_replication(rng):
    return rng.standard_cauchy(size=(N_DRAWS, M_MAX))   # (N_DRAWS, M_MAX)


def quantities_for_slice(theta_slice):     # (n, M)
    by_chain = theta_slice.T               # (M, n)
    pooled = theta_slice.reshape(-1)
    return {
        "RMSE E[theta]":
            (pooled.mean() - TARGET_M1) ** 2,
        "RMSE E[theta^2]":
            ((pooled**2).mean() - TARGET_M2) ** 2,
        "RMSE E[theta^3]":
            ((pooled**3).mean() - TARGET_M3) ** 2,
        "ME R-hat - 1 (theta)":
            rhat_safe(by_chain),
        "ME R-hat - 1 (theta^2)":
            rhat_safe(by_chain**2),
        "ME R-hat - 1 (theta^3)":
            rhat_safe(by_chain**3),
        "RMSE Stein tanh(theta/0.5)":
            stein_error(tanh_stein_accumuland(theta_slice, 0.5)) ** 2,
        "RMSE Stein tanh(theta)":
            stein_error(tanh_stein_accumuland(theta_slice, 1.0)) ** 2,
        "RMSE Stein tanh(theta/2)":
            stein_error(tanh_stein_accumuland(theta_slice, 2.0)) ** 2,
    }


def simulate():
    rng = np.random.default_rng(BASE_SEED)
    records = []
    for rep in range(N_REPLICATIONS):
        print(f"Replication {(rep + 1):6d} / {N_REPLICATIONS:6d}")
        theta = run_replication(rng)
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


def plot_results(agg, outfile="stein_grid_cauchy.png"):
    label_to_row = {label: row for (label, row, _c, _t, _s) in FACET_SPEC}
    df = agg.copy()
    df["row"] = df["quantity"].map(label_to_row)

    is_error = df["row"] == "error"
    is_rhat  = df["row"] == "rhat"
    is_stein = df["row"] == "stein"
    df.loc[is_error, "value"] = np.sqrt(df.loc[is_error, "value"])
    df.loc[is_stein, "value"] = np.sqrt(df.loc[is_stein, "value"])
    df.loc[is_rhat,  "value"] = df.loc[is_rhat, "value"] - 1.0

    df = df[np.isfinite(df["value"]) & (df["value"] > 0)]

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
                title="Cauchy(0,1) target, Cauchy init (independent sampler)")
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
