import os
import pandas as pd
import matplotlib.pyplot as plt

# === КОНФІГУРАЦІЯ ГРАФІКІВ ===
PLOTS_CONFIG = {
    'ApparentMagnitude': {
        'title': 'Крива блиску (Зоряна величина)', 
        'ylabel': 'Apparent Magnitude (m)', 
        'color': 'b', 
        'invert_y': True,
        'filename': 'magnitude_curve.png'
    },
    'Distance_AU': {
        'title': 'Відстань між Землею та кометою', 
        'ylabel': 'Distance (AU)', 
        'color': 'g', 
        'invert_y': False,
        'filename': 'distance_curve.png'
    },
    'PhaseAngle_deg': {
        'title': 'Фазовий кут', 
        'ylabel': 'Phase Angle (deg)', 
        'color': 'r', 
        'invert_y': False,
        'filename': 'phase_angle_curve.png'
    }
}

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, 'photometry_log.csv')
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Помилка: Файл {csv_file} не знайдено.")
        return

    # Фільтрація некоректних даних
    if 'ApparentMagnitude' in df.columns:
        df = df[df['ApparentMagnitude'] < 90.0]

    if df.empty:
        print("Немає валідних даних для малювання графіків.")
        return

    # Генерація окремих файлів для кожної змінної
    for column, settings in PLOTS_CONFIG.items():
        if column not in df.columns:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(df['JD'], df[column], marker='.', linestyle='-', color=settings['color'], markersize=4)
        
        plt.title(settings['title'])
        plt.xlabel('Julian Date (JD)')
        plt.ylabel(settings['ylabel'])
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if settings.get('invert_y', False):
            plt.gca().invert_yaxis()

        plt.tight_layout()
        
        out_img = os.path.join(script_dir, settings['filename'])
        plt.savefig(out_img, dpi=300)
        plt.close() # Очищення пам'яті та підготовка до наступного графіка
        print(f"Збережено графік: {out_img}")

if __name__ == "__main__":
    main()