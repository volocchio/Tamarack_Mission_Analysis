import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, os.pardir, 'simulation.py')
PATH = os.path.abspath(PATH)

try:
    with open(PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
except Exception as e:
    print(f'ERROR: failed to read {PATH}: {e}')
    sys.exit(1)

changed = False

for i in range(len(lines) - 6):
    if re.match(r"^\s*if segment == 6 and last_segment != 6:\s*$", lines[i]):
        # Expect next line to be leveloff += 1
        if not re.match(r"^\s*leveloff \+= 1\s*$", lines[i+1] if i+1 < len(lines) else ''):
            continue
        # If already patched, skip
        if re.match(r"^\s*if climb_fuel_flag == 0:\s*$", lines[i+2] if i+2 < len(lines) else ''):
            # Already patched at this location
            continue
        # Check the four assignment lines
        l2 = lines[i+2] if i+2 < len(lines) else ''
        l3 = lines[i+3] if i+3 < len(lines) else ''
        l4 = lines[i+4] if i+4 < len(lines) else ''
        l5 = lines[i+5] if i+5 < len(lines) else ''
        if not (re.match(r"^\s*climb_fuel = fuel_burned\s*$", l2)
                and re.match(r"^\s*climb_time = t\s*$", l3)
                and re.match(r"^\s*climb_fuel_flag = 1\s*$", l4)
                and re.match(r"^\s*climb_dist = dist_ft / 6076\.12\s*$", l5)):
            continue
        indent = re.match(r"^(\s*)", l2).group(1)
        # Build replacement block
        block = [
            lines[i],               # if segment == 6 and last_segment != 6:
            lines[i+1],             # leveloff += 1
            f"{indent}if climb_fuel_flag == 0:\n",
            f"{indent}    climb_fuel = fuel_burned\n",
            f"{indent}    climb_time = t\n",
            f"{indent}    climb_fuel_flag = 1\n",
            f"{indent}    climb_dist = dist_ft / 6076.12\n",
        ]
        # Splice
        before = lines[:i]
        after = lines[i+6:]
        lines = before + block + after
        changed = True
        break

if changed:
    try:
        with open(PATH, 'w', encoding='utf-8', newline='') as f:
            f.writelines(lines)
        print('Patched simulation.py successfully.')
    except Exception as e:
        print(f'ERROR: failed to write {PATH}: {e}')
        sys.exit(1)
else:
    print('No changes applied (pattern not found or already patched).')
