import numpy as np
import pandas as pd
import math
import copy
import random
import os
from typing import List
import itertools
from functools import lru_cache
from typing import List

outputdir = "./output/"

# パラメータ設定
size = 10 #盤面の大きさ
window = 30 #文字数
roop = 100000 #一度に読み出す文字列長

class IllustLogic:
    # 問題データ（hint_row,hint_col）はクラス引数で引き受けるよう修正
    def __init__(self, size, row, col, sample=False):
        self.size = size
        self.board = np.full([size, size], 'w')
        self.str_w = '.'
        self.str_c = '-'
        self.str_b = 'o'
        self.hint_row = row
        self.hint_col = col

        # 標準入力は受け取らないためコメントアウト
        """
        if sample:
            self.hint_row = [[1], [1], [5], [1], [1]]
            self.hint_col = [[1], [3], [1, 1, 1], [1], [1]]
        else:
            print('Input row numbers (split with spaces):')
            self.hint_row = self.input_hint()
            print('Input column numbers (split with spaces):')
            self.hint_col = self.input_hint()
        """

    def input_hint(self):
        ans = list()
        for i in range(self.size):
            hint_i = list(map(int, input().split()))
            sum_i = sum(hint_i) + len(hint_i) - 1
            if sum_i > self.size:
                raise(Exception('Input numbers are too large.'))
            ans.append(hint_i)
        return ans

    def convert_board_to_str(self):
        board_str = copy.copy(self.board)
        np.place(board_str, board_str == 'w', self.str_w)
        np.place(board_str, board_str == 'c', self.str_c)
        np.place(board_str, board_str == 'b', self.str_b)
        ans = ''
        ans += '+' + '-' * (self.size * 2) + '\n'
        for i in range(self.size):
            ans += '| ' + ' '.join(board_str[i, :]) + '\n'
        return ans

    def convert_hint_to_str(self, hint):
        ans = '\n'.join([' '.join(map(str, h)) for h in hint])
        return(ans)

    def __str__(self):
        ans = 'Board:\n'
        ans += self.convert_board_to_str()
        ans += '\n\nRow hints:\n'
        ans += self.convert_hint_to_str(self.hint_row)
        ans += '\n\nColumn hints:\n'
        ans += self.convert_hint_to_str(self.hint_col)
        return ans

    def combn_cell_pattern(self, n_cell, n_gap):
        if n_cell == 0:
            return [[0] * n_gap]
        if n_gap == 1:
            return [[n_cell]]
        ans = []
        for n_cell_next in range(n_cell + 1):
            prev_ans = self.combn_cell_pattern(n_cell_next, n_gap - 1)
            ans += [[n_cell - n_cell_next] + ans_i for ans_i in prev_ans]
        return ans

    def combn_cross_cell(self, hint):
        n_cell = self.size - sum(hint) - len(hint) + 1
        n_gap = len(hint) + 1
        return self.combn_cell_pattern(n_cell, n_gap)

    def create_all_pattern(self, hint):
        n_cross = self.combn_cross_cell(hint)
        ans = []
        for i in range(len(n_cross)):
            ans_i = []
            for j in range(len(n_cross[i]) - 1):
                if j > 0:
                    ans_i += ['c']
                ans_i += ['c'] * n_cross[i][j]
                ans_i += ['b'] * hint [j]
            ans_i += ['c'] * n_cross[i][-1]
            ans.append(ans_i)
        return ans

    def compare_pattern(self, board, pattern):
        pattern0 = ['w' if b == 'w' else p for b, p in zip(board, pattern)]
        return board == pattern0

    def find_intersect(self, pattern):
        fixed_pat = ['w'] * self.size
        for i in range(self.size):
            pat_i = [p[i] for p in pattern]
            uniq_pat_i = list(set(pat_i))
            if len(uniq_pat_i) == 1:
                fixed_pat[i] = uniq_pat_i[0]
        return fixed_pat

    def solve(self, does_print=True):
        #print('Creating all patterns...')
        all_row_pattern = [self.create_all_pattern(h) for h in self.hint_row]
        all_col_pattern = [self.create_all_pattern(h) for h in self.hint_col]

        #print('Drawing board...')
        n_iter = 0
        while True:
            n_iter += 1
            board_prev = copy.copy(self.board)

            # find fixed cells
            all_row_fixed = [self.find_intersect(p) for p in all_row_pattern]
            all_col_fixed = [self.find_intersect(p) for p in all_col_pattern]
            board_fixed_row = np.array(all_row_fixed)
            board_fixed_col = np.array(all_col_fixed).transpose()

            # draw fixed cells
            new_board = copy.copy(board_fixed_row)
            new_board[new_board == 'w'] = board_fixed_col[new_board == 'w']
            self.board = copy.copy(new_board)

            # finish
            if np.array_equal(self.board, board_prev):
                # 出力の表示はしない
                """
                print(f'iter={n_iter}')
                print(self.convert_board_to_str())
                print('Finished!')
                """
                return self.convert_board_to_str()

            # 出力の表示はしない
            """
            # print process
            if does_print:
                print(f'iter={n_iter}')
                print(self.convert_board_to_str())
            """

            # remove patterns which do not match with present board
            for i in range(self.size):
                all_row_pattern[i] = [
                    p for p in all_row_pattern[i]
                    if self.compare_pattern(board=list(self.board[i, :]), pattern=p)]
                all_col_pattern[i] = [
                    p for p in all_col_pattern[i]
                    if self.compare_pattern(board=list(self.board[:, i]), pattern=p)]

# 数列を再起的に分割し、3次元リストに変換する関数
def split_list(x: List[int], n: int) -> List[List[List[int]]]:
    # x をタプルに変換して変更不可にしておく
    x_tuple = tuple(x)

    @lru_cache(maxsize=None)
    def rec(start: int, parts: int) -> List[List[List[int]]]:
        if parts == 1:
            # 残り全体を1つの部分集合として返す
            return [[list(x_tuple[start:])]]
        
        result = []
        # 分割位置は、残りのparts個の部分集合が作れるように制限
        for i in range(start + 1, len(x_tuple) - parts + 2):
            first_part = list(x_tuple[start:i])
            for tail in rec(i, parts - 1):
                result.append([first_part] + tail)
        return result

    return rec(0, n)

# 数列を、IllustLogicクラスが受け付ける配列に変換する関数
def make_hint(x, n, m):
    # 入力文字列を1度だけ整数リストに変換
    x_list = [int(c) for c in x]
    patterns = split_list(x_list, n)
    rel = []
    for pattern in patterns:
        valid = True
        # 各部分列について1回のループで両方の条件をチェック
        for sub in pattern:
            if sub[0] == 0 or sub[-1] == 0:
                valid = False
                break
            # 部分列の値の計算（各行(列)が盤面サイズ以内であるかのチェック）
            val = sum(sub) + len(sub) - 1 - sub.count(0)
            if val > m:
                valid = False
                break
        if valid:
            rel.append(pattern)
    return rel

# 結果に含まれる"o"の数を、問題データと同じ2次元配列に変換する関数
def count_o(input_str):
  input_str = input_str.replace(" ","")
  return [len(list(group)) for char, group in itertools.groupby(input_str) if char == 'o']

# 二次元配列から0を除く関数
def remove_zero(arr):
  return [[x for x in sublist if x != 0] for sublist in arr]

# 問題データと結果が一致するか判定する関数
def check_answer(row,col,rel):
  rel = list(map(lambda x:x.replace(" ",""),rel.split("\n")[1:-1]))
  row_count = list(map(count_o,rel))
  col_count = list(map(count_o,list(map(lambda x:"".join(x),zip(*rel)))[1:]))

  if( (row_count == remove_zero(row)) and (col_count == remove_zero(col))):
    return True
  else:
    return False

# 考えられる問題データを全通り試行する関数
def check_illst(a,b,size,rows,cols,point):
  best_rel = ""
  best_score = 9999999999

  p = 1

  for row in rows:
    for col in cols:

      ill = IllustLogic(size,row,col)
      z = ill.solve()
      score = z.count(".")
      best_score = min(best_score,score)

      if( (score == 0) and (point == z.count("o")) and (check_answer(row,col,z))):
        filepath = outputdir + "rel_{}_{}_pattern{}.txt".format(a,b,p)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)

        with open(filepath,"w") as f:
          f.write("row:" + str(row) + "\n")
          f.write("col:" + str(col) + "\n")
          f.write(z)

        p += 1

  return best_score

def main():

  with open(outputdir + "log.csv","w") as f:
    f.write("start,split,num_row,num_col,num_pattern,best_score\n")


  for start in range(0,1000000000,roop):

    with open("./pi-10oku.txt","r") as f:
      f.seek(start+2)
      pi = f.read(roop+window)

    # 円周率のうち、解析に使用する部分のみpi_l配列に読み出し
    pi_l = list(map(int,pi))
    
    #for i in pi:
    #  pi_l.append(int(i))

    # 累積和の計算
    n = len(pi_l)
    prefix = [0] * (n+1)
    for idx in range(n):
        prefix[idx+1] = prefix[idx] + pi_l[idx]

    log = []

    for i in range(roop): # 読み出し位置のループ
      for j in range(size,window): # 切り分け位置のループ


        sum_first = prefix[i+j] - prefix[i] #前半部分の総和
        sum_second = prefix[i+window] - prefix[i+j] #後半部分の総和

        # 切り分け位置の前後で合計値が一致したら、問題の判定に入る
        if( sum_first == sum_second):

          rows = make_hint(pi_l[i:i+j],size,size) # 考えれる行方向の組み合わせ
          cols = make_hint(pi_l[i+j:i+window],size,size) # 考えられる列方向の組み合わせ
          point = sum_first # 数値の合計（塗り潰されるマス数）

          # 行方向、列方向それぞれで問題として成立
          if(len(rows) > 0 and len(cols) > 0):
            best_score = check_illst(i,j,size,rows,cols,point)
          else:
            best_score = None

          # ログ出力用のデータの記録
          log.append([start+i,j,len(rows),len(cols),len(rows)*len(cols),best_score])

        # 切り分け位置の前後で合計値が一致しないことが確定したらループを抜ける
        elif( sum_first > sum_second):
          break

    # ログファイルに書き出し
    with open(outputdir + "log.csv","a") as f:
      for i in log:
        f.write(str(i)[1:-1] + "\n")


if __name__ =='__main__':
    main()
