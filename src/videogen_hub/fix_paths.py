import argparse
import os
import re


def find_python_files(directory):
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def compute_absolute_import(base_package, file_path, relative_import):
    depth = file_path.count(os.sep) - os.path.abspath(base_package).count(os.sep)
    if relative_import.startswith("."):
        leading_dots = len(relative_import) - len(relative_import.lstrip('.'))
        absolute_import = base_package + "." + file_path.replace(os.sep, ".").split(".", 1)[1]
        filename = os.path.basename(file_path)
        absolute_import = absolute_import.replace(filename, "")
        if leading_dots > 1:
            for _ in range(leading_dots - 1):
                absolute_import = absolute_import.rsplit('.', 1)[0]
        absolute_import = absolute_import + "." + relative_import.lstrip('.')
    else:
        absolute_import = base_package + "." + relative_import
    absolute_import = absolute_import.replace("..", ".")
    return absolute_import


def replace_relative_imports(file_path, base_package, replace=False):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    new_content = []
    modified = False

    for line in content:
        match = re.match(r'^\s*from\s+(\.+)([^\s]*)\s+import', line)
        if match:
            relative_import = match.group(1) + match.group(2)
            absolute_import = compute_absolute_import(base_package, file_path, relative_import)
            new_line = line.replace(f'from {relative_import} import', f'from {absolute_import} import')
            new_line = new_line.replace("videogen_hub.dev.VideoGenHub.src.", "")
            new_content.append(new_line)
            modified = True
            print(f"File: {file_path}")
            print(f"Old: {line.strip()}")
            print(f"New: {new_line.strip()}\n")
        else:
            new_content.append(line)

    if replace and modified:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(new_content)


def main(directory, replace):
    base_package = os.path.basename(directory)
    python_files = find_python_files(directory)

    for file_path in python_files:
        try:
            replace_relative_imports(file_path, base_package, replace)
        except Exception as e:
            print(f"Error processing file: {file_path}")
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replace relative imports with absolute imports.")
    parser.add_argument('--replace', action='store_true', help="If set, actually replace the imports in the files.")
    args = parser.parse_args()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    main(current_directory, args.replace)
