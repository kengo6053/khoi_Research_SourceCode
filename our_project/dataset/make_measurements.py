import csv
import json
import os

# 入力ファイルと出力フォルダの名前を指定
input_file = '${WORK_DIR}/our_project/dataset/train_dataset/train_test/EX00PM06/VehicleMove.csv'
output_folder = '${WORK_DIR}/our_project/dataset/train_dataset/train_test/EX00PM06/measurements'

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# JSONファイルを1行ずつ出力
with open(input_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for index, row in enumerate(csv_reader):
        # 1つ目と2つ目の値を取り出す
        velocity = float(row[0])
        acceleration = float(row[1])
        
        # 辞書形式でデータを作成
        data = {
            "velocity": velocity,
            "acceleration": acceleration
        }
        
        # ファイル名を作成
        file_name = f"{index:04}.json"
        file_path = os.path.join(output_folder, file_name)
        
        # JSONファイルに書き込む
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

print(f"JSONファイルがフォルダ {output_folder} に出力されました。")