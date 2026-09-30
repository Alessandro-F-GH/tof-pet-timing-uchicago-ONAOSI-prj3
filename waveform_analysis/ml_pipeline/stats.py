from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from utils_fit import fit_ctr_ps

ctr_estimate = fit_ctr_ps


def rmse_ps(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


@dataclass(frozen=True)
class PairedCTRImprovement:
    corrected_ctr_ps: float
    led_ctr_ps: float
    improvement_ps: float
    improvement_fraction: float
    improvement_percent: float
    corrected_rmse_ps: float
    led_rmse_ps: float
    rmse_improvement_ps: float
    rmse_improvement_fraction: float
    rmse_improvement_percent: float


@dataclass(frozen=True)
class PairedReplicaSummary:
    n_pairs: int
    reference_mean: float
    candidate_mean: float
    difference_mean: float
    difference_std: float
    ci_low: float
    ci_high: float


def paired_ctr_improvement(corrected_ps: np.ndarray,led_ps: np.ndarray,fit_cfg: dict,*,seed: int | None = None,led_ctr_ps: float | None = None,led_rmse_ps: float | None = None) -> PairedCTRImprovement:
    corrected=np.asarray(corrected_ps,dtype=np.float64).reshape(-1);led=np.asarray(led_ps,dtype=np.float64).reshape(-1)
    if corrected.shape!=led.shape:raise ValueError("paired comparison requires equal-length residual arrays")
    finite=np.isfinite(corrected)&np.isfinite(led);corrected=corrected[finite];led=led[finite]
    if corrected.size<5:raise ValueError("paired comparison requires at least 5 finite event pairs")
    corrected_point=ctr_estimate(corrected,fit_cfg,seed=seed,bootstrap=False)
    led_ctr=float(ctr_estimate(led,fit_cfg,seed=seed,bootstrap=False).ctr_ps) if led_ctr_ps is None else float(led_ctr_ps)
    ctr_improvement=float(led_ctr-corrected_point.ctr_ps);ctr_fraction=ctr_improvement/led_ctr if led_ctr!=0 else float("nan")
    corrected_rmse=rmse_ps(corrected);led_rmse=float(rmse_ps(led) if led_rmse_ps is None else led_rmse_ps)
    rmse_improvement=float(led_rmse-corrected_rmse);rmse_fraction=rmse_improvement/led_rmse if led_rmse!=0 else float("nan")
    return PairedCTRImprovement(float(corrected_point.ctr_ps),led_ctr,ctr_improvement,float(ctr_fraction),float(100.0*ctr_fraction),corrected_rmse,led_rmse,rmse_improvement,float(rmse_fraction),float(100.0*rmse_fraction))


def paired_replica_difference(reference_values: np.ndarray,candidate_values: np.ndarray,*,seed: int = 1729,n_bootstrap: int = 5000,confidence: float = 0.90) -> PairedReplicaSummary:
    reference=np.asarray(reference_values,dtype=np.float64).reshape(-1);candidate=np.asarray(candidate_values,dtype=np.float64).reshape(-1)
    if reference.shape!=candidate.shape:raise ValueError("paired replica arrays must have the same shape")
    finite=np.isfinite(reference)&np.isfinite(candidate);reference=reference[finite];candidate=candidate[finite]
    if not reference.size:raise ValueError("paired replica comparison requires at least one finite pair")
    if not 0.0<confidence<1.0:raise ValueError("confidence must lie in (0, 1)")
    if n_bootstrap<1:raise ValueError("n_bootstrap must be >= 1")
    differences=reference-candidate;rng=np.random.default_rng(int(seed));draw=rng.integers(0,differences.size,size=(int(n_bootstrap),differences.size));bootstrap_means=differences[draw].mean(axis=1);alpha=1.0-float(confidence)
    return PairedReplicaSummary(int(differences.size),float(np.mean(reference)),float(np.mean(candidate)),float(np.mean(differences)),float(np.std(differences,ddof=1)) if differences.size>1 else 0.0,float(np.quantile(bootstrap_means,alpha/2.0)),float(np.quantile(bootstrap_means,1.0-alpha/2.0)))


def residual_summary(values_ps: np.ndarray) -> dict[str, float | int]:
    values=np.asarray(values_ps,dtype=np.float64).reshape(-1);finite=values[np.isfinite(values)]
    if finite.size==0:return {"n_total":int(values.size),"n_finite":0,"rmse_ps":float("nan"),"mean_ps":float("nan"),"std_ps":float("nan"),"min_ps":float("nan"),"max_ps":float("nan"),"q01_ps":float("nan"),"q99_ps":float("nan")}
    return {"n_total":int(values.size),"n_finite":int(finite.size),"rmse_ps":rmse_ps(finite),"mean_ps":float(np.mean(finite)),"std_ps":float(np.std(finite)),"min_ps":float(np.min(finite)),"max_ps":float(np.max(finite)),"q01_ps":float(np.quantile(finite,0.01)),"q99_ps":float(np.quantile(finite,0.99))}


def format_residual_summary(summary: dict[str, float | int]) -> str:
    return (f"n={summary['n_finite']}/{summary['n_total']} | RMSE={summary['rmse_ps']:.3f} ps | mean={summary['mean_ps']:.3f} ps | "f"std={summary['std_ps']:.3f} ps | q01={summary['q01_ps']:.3f} ps | q99={summary['q99_ps']:.3f} ps | min={summary['min_ps']:.3f} ps | max={summary['max_ps']:.3f} ps")
