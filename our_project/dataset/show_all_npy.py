import numpy as np

# 全要素を表示（書き込み）可能にするための設定
np.set_printoptions(threshold=np.inf)

# output_waypoints.npyから配列を読み込む
# allow_pickle=True を追加
array = np.load('${WORK_DIR}/our_project/dataset/PM00/point_cloud/0000.npy', allow_pickle=True)

# floatを通常表記で出力するためのフォーマッタを指定
formatter = {'float_kind': lambda x: repr(float(x))}

# 配列を文字列化（numpyの配列を全要素文字列化する）
array_str = np.array2string(array, separator=', ', formatter=formatter)

# 行ごとに分割
lines = array_str.split('\n')

# 4行ごとに空行（改行）を挿入
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)

# 再度結合
final_str = '\n'.join(new_lines)

# ファイルに書き込み
with open('${WORK_DIR}/our_project/dataset/PM00/point_cloud/output.txt', 'w') as f:
    f.write(final_str)
