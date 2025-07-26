import os
import json
import argparse

def count_velocity_categories(root_folder, output_folder):
    # カテゴリごとのカウント結果を保存する辞書
    results = {}
    total_counts = {'60-65': 0, '65-70': 0, '70-75': 0, '75以上': 0}  # 合計値の初期化
    
    # 各親フォルダを探索
    for parent_folder in os.listdir(root_folder):
        parent_path = os.path.join(root_folder, parent_folder)
        if not os.path.isdir(parent_path):  # フォルダでない場合はスキップ
            continue
        
        # measurementsフォルダのパス
        measurements_folder = os.path.join(parent_path, 'measurements')
        if not os.path.exists(measurements_folder):
            continue
        
        # velocityのカテゴリごとのカウント
        count_60_65 = 0
        count_65_70 = 0
        count_70_75 = 0
        count_75_above = 0

        # measurementsフォルダ内のJSONファイルを処理
        for file in os.listdir(measurements_folder):
            file_path = os.path.join(measurements_folder, file)
            if not file.endswith('.json'):  # JSONファイルでない場合はスキップ
                continue
            
            # JSONファイルを読み込む
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    velocity = data.get('velocity', None)
                    if velocity is None:
                        continue
                    
                    # velocityをカテゴリごとにカウント
                    if 60 <= velocity < 65:
                        count_60_65 += 1
                    elif 65 <= velocity < 70:
                        count_65_70 += 1
                    elif 70 <= velocity < 75:
                        count_70_75 += 1
                    elif velocity >= 75:
                        count_75_above += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # 親フォルダの結果を保存
        results[parent_folder] = {
            '60-65': count_60_65,
            '65-70': count_65_70,
            '70-75': count_70_75,
            '75以上': count_75_above
        }
        
        # 合計に加算
        total_counts['60-65'] += count_60_65
        total_counts['65-70'] += count_65_70
        total_counts['70-75'] += count_70_75
        total_counts['75以上'] += count_75_above
    
    # 出力フォルダが存在しない場合は作成
    os.makedirs(output_folder, exist_ok=True)
    
    # 結果をテキストファイルに保存
    output_file = os.path.join(output_folder, 'velocity_counts.txt')
    with open(output_file, 'w') as f:
        for parent_folder, counts in results.items():
            f.write(f"親フォルダ: {parent_folder}\n")
            f.write(f"  60-65: {counts['60-65']}個\n")
            f.write(f"  65-70: {counts['65-70']}個\n")
            f.write(f"  70-75: {counts['70-75']}個\n")
            f.write(f"  75以上: {counts['75以上']}個\n")
            f.write("\n")
        
        # 合計値の出力
        f.write("全親フォルダの合計:\n")
        f.write(f"  60-65: {total_counts['60-65']}個\n")
        f.write(f"  65-70: {total_counts['65-70']}個\n")
        f.write(f"  70-75: {total_counts['70-75']}個\n")
        f.write(f"  75以上: {total_counts['75以上']}個\n")
    
    print(f"結果を {output_file} に保存しました。")

if __name__ == "__main__":
    # 引数の設定
    parser = argparse.ArgumentParser(description="JSONファイルのvelocity値を集計")
    parser.add_argument('--root_folder', type=str, help="複数の親フォルダをまとめているルートフォルダのパス")
    parser.add_argument('--output_folder', type=str, help="結果を出力するフォルダのパス")
    args = parser.parse_args()
    
    # 関数の実行
    count_velocity_categories(args.root_folder, args.output_folder)
