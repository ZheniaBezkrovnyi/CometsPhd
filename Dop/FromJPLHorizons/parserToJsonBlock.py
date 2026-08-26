import json

def parse_horizons_to_json(file_path):
    in_data_block = False
    results = []

    # Читаємо файл
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stripped_line = line.strip()
            
            # Шукаємо початок блоку ефемерид
            if stripped_line == '$$SOE':
                in_data_block = True
                continue
            
            # Зупиняємось на кінці блоку
            if stripped_line == '$$EOE':
                break
                
            # Збираємо всі рядки з даними
            if in_data_block and stripped_line:
                # Розбиваємо по комах
                cols = [col.strip() for col in stripped_line.split(',')]
                try:
                    # Індекси згідно з форматом JPL "GEOMETRIC osculating elements"
                    orbit_data = {
                        "JD_Time": float(cols[0]), # Допоміжне поле
                        "Date": cols[1],           # Допоміжне поле
                        "a": float(cols[11]),
                        "e": float(cols[2]),
                        "i": float(cols[4]),
                        "Omega": float(cols[5]),
                        "w": float(cols[6]),
                        "M0": float(cols[9]),
                        "epoch": float(cols[0])
                    }
                    results.append(orbit_data)
                except (IndexError, ValueError) as e:
                    print(f"Помилка парсингу рядка: {stripped_line}\nСуть: {e}")

    if not results:
        print("Дані не знайдено. Перевір, чи ввімкнено 'CSV format' у JPL Horizons.")
        return

    # Виводимо всі знайдені блоки
    for idx, res in enumerate(results):
        print(f"--- Блок #{idx + 1} | Дата: {res['Date']} (JD: {res['JD_Time']}) ---")
        
        # Видаляємо допоміжні поля перед виводом фінального JSON
        del res['Date']
        del res['JD_Time']
        
        # Друкуємо готовий шматок для твого конфігу
        print(json.dumps(res, indent=4))
        print("\n")

if __name__ == "__main__":
    # ТУТ ВКАЗУЄТЬСЯ ШЛЯХ ДО ФАЙЛУ
    # Зміни шлях на той, де лежить твій завантажений .txt
    file_path = 'C:/Users/Yevhen/Projects/Univ/Dop/FromJPLHorizons/17_08_2026/m_Bennu.txt' 
    parse_horizons_to_json(file_path)