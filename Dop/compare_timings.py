import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / 'runs'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK = '#3d3d3a'
GRID = '#e4e3dd'
LOG_RATIO = 20.0     # якщо max/min більше - вісь логарифмічна
GPU_ZERO_MS = 0.01   # GPU-час на кадр нижче цього скрізь - блок без CUDA (OpenGL), не малюємо


def find_runs(args):
    """Аргумент - папка прогону, папка з прогонами (наприклад runs) або шматок назви папки в runs/."""
    found = []
    for arg in args:
        path = Path(arg)
        candidates = [path] if path.exists() else sorted(RUNS.glob(f'*{arg}*'))
        if not candidates:
            print(f"Не знайдено: {arg}")
        for c in candidates:
            if (c / 'timings.csv').is_file():
                found.append(c)
            elif c.is_dir():
                found += sorted(d for d in c.iterdir() if (d / 'timings.csv').is_file())
    unique, seen = [], set()
    for r in found:
        key = r.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def face_count(run):
    """Кількість граней з заголовка PLY, на який посилається конфіг у папці прогону."""
    for cfg in run.glob('*.json'):
        try:
            model = json.loads(cfg.read_text(encoding='utf-8')).get('modelPath')
        except (ValueError, OSError):
            continue
        if not model:
            continue
        ply = Path(model) if Path(model).is_absolute() else REPO / model
        try:
            with open(ply, 'rb') as f:
                for raw in f:
                    line = raw.decode('latin-1').strip()
                    m = re.match(r'element\s+face\s+(\d+)', line)
                    if m:
                        return int(m.group(1))
                    if line == 'end_header':
                        break
        except OSError:
            pass
    return None


def short_label(run):
    return re.sub(r'^\d{4}-\d{2}-\d{2}_\d{6}_', '', run.name) or run.name


def count_text(n):
    if n >= 1e6:
        return f'{n / 1e6:.3g} млн'
    if n >= 1e3:
        return f'{n / 1e3:.3g} тис.'
    return f'{n:.0f}'


def duration_text(sec):
    if not np.isfinite(sec):
        return '-'
    if sec < 1:
        return f'{sec * 1000:.3g} мс'
    if sec < 60:
        return f'{sec:.3g} с'
    if sec < 3600:
        return f'{int(sec // 60)} хв {int(sec % 60)} с'
    return f'{int(sec // 3600)} год {int(sec % 3600 // 60)} хв'


def time_unit(max_sec):
    """Одиниця осі Y під найбільше значення, щоб числа були читабельні."""
    if max_sec >= 2 * 3600:
        return 3600.0, 'години'
    if max_sec >= 120:
        return 60.0, 'хвилини'
    if max_sec >= 1:
        return 1.0, 'секунди'
    return 0.001, 'мілісекунди'


def wants_log(values):
    v = np.asarray([x for x in values if np.isfinite(x) and x > 0])
    return len(v) > 1 and v.max() / v.min() > LOG_RATIO


def main():
    parser = argparse.ArgumentParser(description='Скільки часу кожен блок рахує за весь прогін, по кількох прогонах')
    parser.add_argument('runs', nargs='+', help='папки прогонів, папка runs або шматок назви')
    parser.add_argument('--log', action='store_true', help='завжди логарифмічні осі')
    parser.add_argument('--linear', action='store_true', help='завжди лінійні осі')
    parser.add_argument('--by-run', action='store_true', help='по X прогони, навіть якщо відома кількість граней')
    parser.add_argument('--no-show', action='store_true', help='тільки зберегти PNG, без вікна')
    args = parser.parse_args()

    runs = find_runs(args.runs)
    if not runs:
        print("Жодного timings.csv не знайдено.")
        sys.exit(1)

    # Коротка назва без дати; якщо дві однакові - лишаємо повну, щоб прогони не злились
    short = [short_label(r) for r in runs]
    names = [s if short.count(s) == 1 else r.name for s, r in zip(short, runs)]

    rows = []
    for run, name in zip(runs, names):
        df = pd.read_csv(run / 'timings.csv')
        df['run'] = name
        df['faces'] = face_count(run)
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)

    faces = data.groupby('run', sort=False)['faces'].first().reindex(names)
    by_faces = (not args.by_run and faces.notna().all() and faces.nunique() == len(names) and len(names) > 1)
    if by_faces:
        order = faces.sort_values().index.tolist()
        x = faces.loc[order].astype(float).to_numpy()
        columns = [count_text(n) for n in x]
    else:
        order = names
        x = np.arange(len(order), dtype=float)
        columns = order

    sections = list(dict.fromkeys(data['section']))
    if len(sections) > len(SERIES):
        print(f"Блоків {len(sections)}, на графіку перші {len(SERIES)}.")
        sections = sections[:len(SERIES)]

    def table(column):
        return data.pivot_table(index='section', columns='run', values=column).reindex(index=sections, columns=order)

    cpu = table('cpu_total_s')
    gpu = table('gpu_total_s')
    gpu_per_frame = table('gpu_ms_per_call')
    total = cpu.sum(axis=0, min_count=1)

    lines = []   # (назва, секунди по прогонах, колір, стиль, маркер)
    for i, sec in enumerate(sections):
        lines.append((f'{sec}, CPU', cpu.loc[sec].to_numpy(), SERIES[i], '-', 'o'))
        if np.nanmax(gpu_per_frame.loc[sec].to_numpy()) >= GPU_ZERO_MS:
            lines.append((f'{sec}, GPU', gpu.loc[sec].to_numpy(), SERIES[i], '--', 's'))
    lines.append(('Усі блоки разом, CPU', total.to_numpy(), INK, '-', 'D'))

    all_sec = np.concatenate([y for _, y, *_ in lines])
    scale, unit = time_unit(np.nanmax(all_sec))
    log_y = args.log or (not args.linear and wants_log(all_sec))
    log_x = by_faces and (args.log or (not args.linear and wants_log(x)))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, y, color, style, marker in lines:
        ax.plot(x, y / scale, color=color, linestyle=style, linewidth=2, marker=marker, markersize=7, label=name)

    if log_y:
        ax.set_yscale('log')
        pos = all_sec[np.isfinite(all_sec) & (all_sec > 0)] / scale
        ax.set_ylim(10 ** np.floor(np.log10(pos.min())), 10 ** np.ceil(np.log10(pos.max())))   # цілі декади з підписами
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    if log_x:
        ax.set_xscale('log')
    if by_faces:
        ax.set_xlabel('кількість граней моделі')
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20, ha='right')
        ax.set_xlabel('прогін')
    ax.set_ylabel(f'час за весь прогін, {unit}' + (' (логарифмічна шкала)' if log_y else ''))
    ax.set_title('Скільки часу кожен блок рахує за весь прогін')
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()

    readable = pd.DataFrame({c: [duration_text(y[j]) for _, y, *_ in lines] for j, c in enumerate(columns)},
                            index=[name for name, *_ in lines])
    print('\nЧас за весь прогін:\n')
    print(readable.to_string())

    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    RUNS.mkdir(exist_ok=True)
    png = RUNS / f'timings_compare_{stamp}.png'
    n = 2
    while png.exists():   # два запуски в одну секунду не перезаписують один одного
        stamp = f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{n}"
        png = RUNS / f'timings_compare_{stamp}.png'
        n += 1
    fig.savefig(png, dpi=200)
    pd.DataFrame({c: [y[j] for _, y, *_ in lines] for j, c in enumerate(columns)},
                 index=[name for name, *_ in lines]).to_csv(RUNS / f'timings_compare_{stamp}.csv', encoding='utf-8-sig')
    print(f"\nЗбережено: {png}")
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
