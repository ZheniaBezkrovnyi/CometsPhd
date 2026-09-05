import os
import pandas as pd
import matplotlib.pyplot as plt

def load_jpl_horizons(filepath):
    dates = []
    mags = []
    in_data_block = False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped == '$$SOE':
                in_data_block = True
                continue
            if stripped == '$$EOE':
                break
                
            if in_data_block and stripped:
                cols = [c.strip() for c in stripped.split(',')]
                if len(cols) > 5:
                    # Дата лежить у першій колонці (напр. '2025-Nov-29 00:00')
                    date_str = cols[0] 
                    # APmag лежить у шостій колонці (індекс 5)
                    try:
                        ap_mag = float(cols[5])
                        dates.append(date_str)
                        mags.append(ap_mag)
                    except ValueError:
                        continue

    # Створюємо DataFrame та конвертуємо дати у Юліанські дні (JD)
    df = pd.DataFrame({'DateStr': dates, 'JPL_Mag': mags})
    df['Datetime'] = pd.to_datetime(df['DateStr'], format='%Y-%b-%d %H:%M')
    
    # Вбудована функція pandas для отримання JD
    df['JD'] = df['Datetime'].apply(lambda x: x.to_julian_date())
    return df

def main():
    # Визначаємо папку, у якій знаходиться сам Python-скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Динамічні шляхи до файлів відносно папки скрипта
    sim_file = os.path.join(script_dir, 'photometry_log.csv')
    
    # os.path.normpath коректно обробляє перехід на рівень вище "../"
    jpl_file = os.path.normpath(os.path.join(script_dir, '../FromJPLHorizons/17_08_2026/m_Bennu.txt'))
    
    # 1. Завантаження симуляції
    try:
        df_sim = pd.read_csv(sim_file)
        df_sim = df_sim[df_sim['ApparentMagnitude'] < 90.0] # Відкидаємо помилки
    except FileNotFoundError:
        print(f"Помилка: Не знайдено файл симуляції {sim_file}")
        return

    # 2. Завантаження даних JPL
    try:
        df_jpl = load_jpl_horizons(jpl_file)
    except FileNotFoundError:
        print(f"Помилка: Не знайдено файл JPL {jpl_file}")
        return

    # 3. Побудова графіка
    plt.figure(figsize=(12, 7))
    
    # Малюємо симуляцію (часті точки, детальна крива)
    plt.plot(df_sim['JD'], df_sim['ApparentMagnitude'], 
             label='Simulation (3D Rotational Lightcurve)', 
             color='blue', linewidth=1.5)
    
    # Малюємо дані JPL Horizons (рідкі точки кожні 10 хв, згладжена лінія)
    plt.plot(df_jpl['JD'], df_jpl['JPL_Mag'], 
             label='JPL Horizons (H-G Model Baseline)', 
             color='red', linestyle='--', linewidth=2)

    # Обмежуємо вісь X межами нашої симуляції (4.3 години)
    min_jd = df_sim['JD'].min()
    max_jd = df_sim['JD'].max()
    plt.xlim(min_jd, max_jd)

    # Інверсія осі Y (для зоряних величин)
    plt.gca().invert_yaxis()

    plt.title('Порівняння видимої зоряної величини: Симуляція vs JPL Horizons')
    plt.xlabel('Julian Date (JD)')
    plt.ylabel('Apparent Magnitude (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Збереження картинки поруч зі скриптом
    out_img = os.path.join(script_dir, 'lightcurve_comparison.png')
    plt.savefig(out_img, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()