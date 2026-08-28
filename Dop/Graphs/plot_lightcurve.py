import pandas as pd
import matplotlib.pyplot as plt

def main():
    csv_file = 'photometry_log.csv' # або просто 'photometry_log.csv' залежно від робочої директорії
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Помилка: Файл {csv_file} не знайдено.")
        return

    # Відкидаємо некоректні значення (наприклад, 99.0, які повертає С++ при помилках видимості)
    df = df[df['ApparentMagnitude'] < 90.0]

    if df.empty:
        print("Немає валідних даних для малювання графіка.")
        return

    # Створення графіка
    plt.figure(figsize=(10, 6))
    plt.plot(df['JD'], df['ApparentMagnitude'], marker='.', linestyle='-', color='b', markersize=4)

    # Інверсія осі Y (астрономічний стандарт: чим яскравіше, тим менша величина)
    plt.gca().invert_yaxis()

    plt.title('Крива блиску (Apparent Magnitude vs Julian Date)')
    plt.xlabel('Julian Date (JD)')
    plt.ylabel('Apparent Magnitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Автоматичне налаштування відступів
    plt.tight_layout()

    # Збереження і показ
    plt.savefig('light_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()