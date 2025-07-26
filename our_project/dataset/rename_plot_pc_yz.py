#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def main():
    if len(sys.argv) < 2:
        print("使い方: python rename_files.py <対象フォルダのパス>")
        sys.exit(1)

    target_folder = sys.argv[1]

    # 指定フォルダ内のファイルを取得
    for filename in os.listdir(target_folder):
        old_path = os.path.join(target_folder, filename)

        # ファイルかどうかチェック
        if os.path.isfile(old_path):
            # 「_yz_view」を削除したファイル名を作成
            new_filename = filename.replace("_yz_view", "")
            new_path = os.path.join(target_folder, new_filename)

            # もし差分があるならリネーム
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    main()
