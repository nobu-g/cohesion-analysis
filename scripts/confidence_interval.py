import sys

import pandas as pd
import math
from scipy import stats


def main():
    df = pd.read_csv(sys.argv[1], sep=',')

    if len(df) == 1:
        print(df)
        return

    columns = []
    uppers = []
    deltas_plus = []
    middles = []
    deltas_minus = []
    lowers = []
    for column, items in df.iteritems():
        columns.append(column)
        n = len(items)
        """
        alpha: 何パーセント信頼区間か
        df: t分布の自由度
        loc: 平均 X bar
        scale: 標準偏差 s
        """
        lower, upper = stats.t.interval(alpha=0.95,
                                        df=n - 1,
                                        loc=items.mean(),
                                        scale=math.sqrt(items.var() / n))
        middle = (upper + lower) / 2
        uppers.append(upper)
        deltas_plus.append(upper - middle)
        middles.append(middle)
        deltas_minus.append(lower - middle)
        lowers.append(lower)

    df_interval = pd.DataFrame.from_dict({'upper': uppers,
                                          'delta+': deltas_plus,
                                          'middle': middles,
                                          'delta-': deltas_minus,
                                          'lower': lowers},
                                         orient='index',
                                         columns=columns)
    print(df_interval)


if __name__ == '__main__':
    main()
