#!/usr/bin/env python3
"""
Порівняння прогону CometsPhd з реальними вимірами кривої блиску.

Запуск (автоматично як postprocess-скрипт або вручну):
    python checks/damit/compare_damit.py <run_dir> [папка_порівняння]

Папка порівняння (data/Comparing/<щось>) визначається так, по черзі:
  1. другий аргумент: ім'я папки (напр. 111_Eros) або повний шлях;
  2. CASE у налаштуваннях нижче, якщо не None;
  3. автоматично за конфігом прогону (App кладе його копію в run_dir):
     спершу шукається конфіг з тим самим ім'ям у COMPARING_DIR/*/,
     якщо нема - конфіг з тими самими фізичними параметрами спостереження
     (startJulianDate, період, полюс, орбіти). Тобто конфіг можна класти в
     Configs під будь-яким ім'ям - папка порівняння знайдеться за вмістом.

Що має лежати в папці порівняння:
  - конфіг *.json, з яким запускалась симуляція;
  - один файл з виміряною кривою (OBSERVED_GLOB): CSV з заголовком і
    колонками OBS_JD_COL та OBS_MAG_COL. JD у тій самій шкалі часу, що й
    JD симуляції (у photometry_log.csv).

Результат кладеться в run_dir: <OUT_PREFIX><папка>.png і .json
"""
import json
import sys
from pathlib import Path

import numpy as np

# ================================ НАЛАШТУВАННЯ ================================
REPO          = Path(__file__).resolve().parents[2]
COMPARING_DIR = REPO / "data" / "Comparing"   # тут папки порівнянь: 111_Eros, 222_..., ...
CASE          = None                          # None = автопошук; або напр. "111_Eros"
OBSERVED_GLOB = "*_observed.csv"              # файл з виміряною кривою в папці порівняння
OBS_JD_COL    = "JD_lt"                       # колонка часу (JD) у файлі вимірів
OBS_MAG_COL   = "mag_rel"                     # колонка відносної зоряної величини
SIM_LOG       = "photometry_log.csv"          # лог фотометрії симуляції в run_dir
OUT_PREFIX    = "compare_"                    # -> compare_<папка>.png / .json
PHASE_SCAN    = 0.1                           # ± частка оберту для підбору фази (0 = вимкнути)
# ==============================================================================


def run_configs(run):
    """Копії конфігу в run_dir (json з розділом physics)."""
    out = []
    for p in sorted(run.glob("*.json")):
        if p.name.startswith(OUT_PREFIX):
            continue
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(cfg, dict) and "physics" in cfg:
            out.append((p, cfg))
    return out


FINGERPRINT = ("startJulianDate", "rotationPeriodHours", "poleRA", "poleDEC", "cometOrbit", "earthOrbit")


def physics_key(cfg):
    ph = cfg.get("physics", {})
    return json.dumps({k: ph.get(k) for k in FINGERPRINT}, sort_keys=True)


def load_json_with_comments(path):
    txt = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("//"))
    return json.loads(txt)


def find_case(run, arg, cfgs):
    if arg:
        p = Path(arg)
        return p if p.is_dir() else COMPARING_DIR / arg
    if CASE:
        return COMPARING_DIR / CASE
    names = [p.name for p, _ in cfgs]
    hits = sorted({m.parent for n in names for m in COMPARING_DIR.glob(f"*/{n}")})
    if not hits:                                            # за вмістом конфігу
        keys = {physics_key(c) for _, c in cfgs}
        for m in COMPARING_DIR.glob("*/*.json"):
            try:
                if physics_key(load_json_with_comments(m)) in keys:
                    hits.append(m.parent)
            except (ValueError, UnicodeDecodeError, AttributeError):
                continue
        hits = sorted(set(hits))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"Не знайшов папку порівняння в {COMPARING_DIR}: немає конфігу з ім'ям {names} "
                         f"або з тими самими параметрами physics. Передай папку другим аргументом або задай CASE.")
    raise SystemExit(f"Підходить кілька папок: {[h.name for h in hits]}. "
                     f"Передай потрібну другим аргументом або задай CASE.")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    cfgs = run_configs(run)
    if not cfgs:
        raise SystemExit(f"У {run} немає копії конфігу (*.json з physics).")
    P_days = cfgs[0][1]["physics"]["rotationPeriodHours"] / 24.0

    case = find_case(run, sys.argv[2] if len(sys.argv) > 2 else None, cfgs)
    if not case.is_dir():
        raise SystemExit(f"Папки порівняння немає: {case}")
    obs_files = sorted(case.glob(OBSERVED_GLOB))
    if len(obs_files) != 1:
        raise SystemExit(f"У {case} має бути рівно один файл {OBSERVED_GLOB}, знайдено: "
                         f"{[f.name for f in obs_files]}")
    obs_csv = obs_files[0]

    sim = np.genfromtxt(run / SIM_LOG, delimiter=",", names=True)
    ok = sim["ApparentMagnitude"] < 90                      # 99 = нічого не видно
    ts, ms = sim["JD"][ok], sim["ApparentMagnitude"][ok]
    obs = np.genfromtxt(obs_csv, delimiter=",", names=True)
    for col in (OBS_JD_COL, OBS_MAG_COL):
        if col not in obs.dtype.names:
            raise SystemExit(f"У {obs_csv.name} немає колонки '{col}'. Є: {list(obs.dtype.names)}")
    to, mo = obs[OBS_JD_COL], obs[OBS_MAG_COL]
    if to[0] < ts[0] or to[-1] > ts[-1]:
        raise SystemExit(f"Симуляція (JD {ts[0]:.5f}-{ts[-1]:.5f}) не покриває виміри "
                         f"(JD {to[0]:.5f}-{to[-1]:.5f}): перевір startJulianDate і durationRotations.")

    def fit(shift_days):
        msi = np.interp(to + shift_days, ts, ms)
        off = np.mean(mo - msi)                              # криві відносні: вільний лише зсув
        r = mo - (msi + off)
        return float(np.sqrt(np.mean(r ** 2))), off, r

    rms0, off0, r0 = fit(0.0)
    sb, rms1, off1 = 0.0, rms0, off0
    if PHASE_SCAN > 0:
        lo, hi = ts[0] - to[0], ts[-1] - to[-1]               # допустимі зсуви
        shifts = np.arange(max(lo, -PHASE_SCAN * P_days), min(hi, PHASE_SCAN * P_days), 10.0 / 86400.0)
        if len(shifts):
            sb = float(shifts[np.argmin([fit(s)[0] for s in shifts])])
            rms1, off1, _ = fit(sb)
    dphi = sb / P_days * 360.0
    m_sim_at_obs = np.interp(to, ts, ms)

    res = {
        "case": case.name, "observed_csv": str(obs_csv), "n_obs": int(len(to)),
        "amp_obs_mag": float(mo.max() - mo.min()),
        "amp_sim_mag": float(m_sim_at_obs.max() - m_sim_at_obs.min()),
        "sim_mean_mag": float(np.mean(m_sim_at_obs)),
        "obs_shift_to_sim_mag": float(-off0),
        "rms_a_priori_mag": rms0,
        "rms_phase_fit_mag": rms1, "phase_fit_deg": dphi, "phase_fit_minutes": sb * 1440.0,
    }
    out = run / f"{OUT_PREFIX}{case.name}"
    out.with_suffix(".json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{OUT_PREFIX}{case.name}]", json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, surf = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    c_obs, c_sim = "#2a78d6", "#eb6834"
    h = lambda t: (t - to[0]) * 24.0
    sel = (ts >= to[0] - 0.01) & (ts <= to[-1] + 0.01)
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=surf)
    ax.set_facecolor(surf)
    ax.grid(True, color=grid, lw=0.8)
    ax.tick_params(colors=muted, labelcolor=ink)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot(h(ts[sel]), ms[sel], color=c_sim, lw=2, label=f"Simulation, a priori (RMS {rms0:.3f})")
    if PHASE_SCAN > 0:
        ax.plot(h(ts[sel] - sb), ms[sel] + off1 - off0, color=c_sim, lw=1.2, ls="--",
                label=f"Simulation, phase {dphi:+.1f} deg (RMS {rms1:.3f})")
    ax.plot(h(to), mo - off0, "o", ms=4.5, mfc=c_obs, mec=surf, mew=0.8,
            label=f"Observed ({obs_csv.name}), relative - shifted to simulation mean")
    ax.set_xlim(h(to[0]) - 0.1, h(to[-1]) + 0.1)
    ax.invert_yaxis()
    ax.set_ylabel("Apparent magnitude, simulation (mag)", color=ink)
    ax.set_xlabel(f"Hours since JD {to[0]:.6f}", color=ink)
    fig.suptitle(f"{case.name}: simulation vs observation", x=0.012, ha="left", color=ink, fontsize=12)
    ax.legend(frameon=False, labelcolor=ink, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"Збережено: {out.with_suffix('.png')}")
    print(f"Збережено: {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
