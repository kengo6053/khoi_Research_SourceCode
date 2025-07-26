import os
import json
import argparse

def find_missing_accel(root_dir):
    missing = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.json'):
                path = os.path.join(dirpath, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'basic_acceleration' not in data or data['basic_acceleration'] is None:
                        missing.append(path)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
    return missing

def main():
    parser = argparse.ArgumentParser(description='Check missing acceleration in JSON files.')
    parser.add_argument('--train_dir', help='Path to the train directory')
    args = parser.parse_args()

    missing = find_missing_accel(args.train_dir)
    if missing:
        print(f"Found {len(missing)} files with missing acceleration:")
        for p in missing:
            print(f"  {p}")
    else:
        print("✅ No missing acceleration values found.")

if __name__ == '__main__':
    main()

