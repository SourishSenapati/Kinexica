import os
import re

print("Starting fixes...")

# 1. Global text replacements
for root, _, files in os.walk('.'):
    if 'venv' in root or '.git' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith(('.py', '.md', '.txt', '.csv', '.swift', '.js', '.html')):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # replacements
            new_content = content.replace("Kinexica", "Kinexica")
            new_content = new_content.replace("Kinexica", "Kinexica")
            new_content = new_content.replace("Kinexica", "Kinexica")
            new_content = new_content.replace("ft.Colors.", "ft.Colors.")

            if new_content != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"Updated global text in {filepath}")

# 2. Fix LICENSE markers
try:
    with open('LICENSE', 'r', encoding='utf-8') as f:
        content = f.read()
    content = re.sub(r'(\d+\.)  ', r'\1 ', content)
    content = re.sub(r'([-\*])  ', r'\1 ', content)
    with open('LICENSE', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed LICENSE MD030")
except BaseException as e:
    print('LICENSE error:', e)

# 3. main.py Unused BaseModel, Import outside toplevel
try:
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace(
        "from pydantic import BaseModel, Field", "from pydantic import Field")
    content = content.replace("from pydantic import BaseModel", "")

    # move import to top level and remove indentation
    has_inference_import = "from pinn_engine.inference import run_inference" in content
    if has_inference_import:
        # Search for exact line with indent
        content = content.replace(
            "    from pinn_engine.inference import run_inference\n", "")
        # Add to top (after docstring or other imports)
        if "from pinn_engine.inference import run_inference" not in content:
            content = "from pinn_engine.inference import run_inference\n" + content

    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed main.py issues")
except BaseException as e:
    print('main.py error:', e)

# 4. pinn_engine/train_pinn.py f-string without interp
try:
    with open('pinn_engine/train_pinn.py', 'r', encoding='utf-8') as f:
        content = f.read()
    # Find f"xyz" and replace with "xyz" if no braces
    content = re.sub(r'f"(Loading[^\"]*)"', r'"\1"', content)
    content = re.sub(r'f\'(Loading[^\']*)\'', r"'\1'", content)
    content = content.replace('f"Calculating', '"Calculating')
    content = content.replace("f'Calculating", "'Calculating")
    content = content.replace('f"Compiling', '"Compiling')
    with open('pinn_engine/train_pinn.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed train_pinn.py f-strings")
except BaseException as e:
    print('train_pinn.py error:', e)

# 5. pinn_engine/data_scraper.py line length
try:
    with open('pinn_engine/data_scraper.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line) > 100 and "cv_variance" in line:
            lines[i] = line.replace(
                "'cv_variance'", "\\\n            'cv_variance'")
            break
    with open('pinn_engine/data_scraper.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Fixed data_scraper.py line length")
except BaseException as e:
    pass

# 6. pinn_engine/inference.py line length
try:
    with open('pinn_engine/inference.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if len(line) > 100 and "expected" in line:
            lines[i] = line.replace(", expected:", ",\\\n        expected:")
            break
    with open('pinn_engine/inference.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Fixed inference.py line length")
except BaseException as e:
    pass

print("All fixes applied!")
