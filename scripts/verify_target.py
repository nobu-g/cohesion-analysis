import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', const='', type=str, nargs='?',
                        help='original target name')
    parser.add_argument('--expr', '-e', required=True, type=str,
                        help='experiment name')
    args = parser.parse_args()

    is_pred_available = '-vpa' in args.expr
    is_noun_available = '-npa' in args.expr

    if '_' in args.target:
        corpus, pas_target = args.target.split('_')
    else:
        corpus, pas_target = args.target, 'none'
    if is_pred_available:
        if is_noun_available:
            if pas_target not in ('pred', 'noun', 'all'):
                pas_target = 'pred'
        else:
            pas_target = 'pred'
    else:
        if is_noun_available:
            pas_target = 'noun'
        else:
            pas_target = 'none'

    if corpus not in ('kwdlc', 'kc', 'fuman', 'all'):
        corpus = 'all'

    print(corpus, end='')
    if pas_target != 'none':
        print('_' + pas_target)


if __name__ == '__main__':
    main()
