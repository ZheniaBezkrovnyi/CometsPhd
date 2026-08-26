import argparse
from pathlib import Path

# Папки, які ШІ не потрібно читати (кеш, збірки, системи контролю версій)
EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode", ".vs", "out", "build", 
    "bin", "obj", "__pycache__", "Release", "Debug",
    "External" 
}

# Специфічні файли, які варто ігнорувати
EXCLUDE_FILES = {
    "_project_dump.txt",
    "CMakeCache.txt"
}

# Розширення, які несуть користь для контексту (C++, CUDA, Python, конфіги)
INCLUDE_EXTENSIONS = {
    ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".glsl",
    ".py", ".json", ".md", ".cmake", ".txt", ".yaml", ".yml"
}

def should_skip_dir(path: Path, root: Path) -> bool:
    # Перевіряємо, чи є частина шляху в списку виключень
    return any(part in EXCLUDE_DIRS for part in path.relative_to(root).parts)

def should_include_file(path: Path, root: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return False
    
    # Спеціальний виняток для CMakeLists.txt
    if path.suffix.lower() not in INCLUDE_EXTENSIONS and path.name != "CMakeLists.txt":
        return False

    if should_skip_dir(path.parent, root):
        return False

    return True

def read_text_file(path: Path) -> str:
    # Обробка кодувань для гарантованого зчитування тексту
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

def main():
    parser = argparse.ArgumentParser(description="Створення дампу проекту для ШІ.")
    parser.add_argument(
        "target_dir", 
        type=str, 
        nargs="?", 
        default="C:/Users/Yevhen/Projects/Univ/CometsPhd", 
        help="Шлях до папки проекту"
    )
    parser.add_argument("-o", "--output", type=str, default="_project_dump.txt", help="Ім'я вихідного файлу (за замовчуванням _project_dump.txt)")
    args = parser.parse_args()

    root_path = Path(args.target_dir).resolve()
    
    # ЗМІНА ТУТ: Отримуємо шлях до папки, де лежить dump_project.py
    script_dir = Path(__file__).resolve().parent
    # Зберігаємо файл у папку зі скриптом
    output_file = script_dir / args.output

    if not root_path.is_dir():
        print(f"Помилка: Шлях {root_path} не існує або не є папкою.")
        return

    # Збираємо файли, ігноруючи непотрібне
    files = sorted(
        path for path in root_path.rglob("*")
        if path.is_file() and should_include_file(path, root_path)
    )

    with output_file.open("w", encoding="utf-8") as out:
        for path in files:
            relative_path = path.relative_to(root_path)

            out.write(f"\n{'=' * 100}\n{relative_path}\n{'=' * 100}\n\n")
            out.write(read_text_file(path))
            out.write("\n\n")

    print(f"Готово. Зібрано файлів: {len(files)}")
    print(f"Дамп збережено у: {output_file}")

if __name__ == "__main__":
    main()