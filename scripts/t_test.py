import argparse
import math
import sys

import pandas as pd
from scipy import stats


def calc_interval(df: pd.DataFrame) -> pd.DataFrame:
    means = []
    deltas = []
    for _, items in df.items():
        n = len(items)
        mean = items.mean()
        var = items.var()
        if var == 0:
            means.append(items.mean())
            deltas.append(0)
            continue
        """
        t分布で信頼区間を計算
        alpha: 何パーセント信頼区間か
        df: t分布の自由度
        loc: 平均 X bar
        scale: 標準偏差 s
        """
        lower, upper = stats.t.interval(alpha=0.95,
                                        df=n - 1,
                                        loc=mean,
                                        scale=math.sqrt(var / n))
        means.append(mean)
        deltas.append(upper - mean)

    return pd.DataFrame.from_dict({'mean': means,
                                   'delta': deltas},
                                  orient='index',
                                  columns=df.columns)


def calc_pval(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    t_vals = []
    p_vals = []
    cols = []
    for col in [col for col in df1.columns if col in df2.columns]:
        """
        Welchのt検定 (サンプル間に対応なし & 等分散性なし)
        """
        t_val, p_val = stats.ttest_ind(df2[col], df1[col], equal_var=False)
        t_vals.append(t_val)
        p_vals.append(p_val)
        cols.append(col)

    return pd.DataFrame.from_dict({'t-value': t_vals, 'p-value': p_vals},
                                  orient='index',
                                  columns=cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, nargs='+',
                        help='1 or 2 csv files to test')
    parser.add_argument('--precision', '--prec', default=4, type=int,
                        help='Number of digits after the decimal point')
    parser.add_argument('--percent', action='store_true', default=False,
                        help='display values as percent (%)')
    args = parser.parse_args()

    pd.options.display.precision = args.precision

    if len(args.data) == 1:
        df_interval = calc_interval(pd.read_csv(args.data[0]))
        print(df_interval.applymap(lambda x: x * (100 if args.percent else 1)))
    elif len(args.data) == 2:
        df1 = pd.read_csv(args.data[0])
        df2 = pd.read_csv(args.data[1])
        print(sys.argv[1] + ':')
        print(calc_interval(df1).applymap(lambda x: x * (100 if args.percent else 1)), end='\n\n')
        print(sys.argv[2] + ':')
        print(calc_interval(df2).applymap(lambda x: x * (100 if args.percent else 1)), end='\n\n')
        pd.options.display.precision = 4
        print(calc_pval(df1, df2))
    else:
        print('error: too many arguments', file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    main()
