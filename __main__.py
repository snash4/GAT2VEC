from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.gat2vec import gat2vec
from Evaluation.Classification import Classification


def main():
    parser = ArgumentParser("gat2vec",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', nargs='?', required=True,
                        help='Input data directory in ../data/ folder')

    parser.add_argument('--label', nargs='?', default=False, type=bool,
                        help=' If data is labelled')

    parser.add_argument('--algo', nargs='?', default='g2v', type=str,
                        help=' Algo to use (gat2vec/gat2vec_bip')

    parser.add_argument('--num-walks', default=10, type=int,
                        help='Random walks per node')

    parser.add_argument('--walk-length', default=80, type=int,
                        help='Random walk length')

    parser.add_argument('--output', default=True,
                        help='save output embedding')

    parser.add_argument('--dimension', default=128, type=int,
                        help='size of representation.')

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of skipgram model.')
    return parser.parse_args()

if __name__ == "__main__":
    args = main()
    g2v = gat2vec(args.data, args.label)
    if args.algo == 'g2v':
        model = g2v.train_gat2vec(args.data,  args.num_walks, args.walk_length, args.dimension,
                                 args.window_size, args.output)
    else:
        model = g2v.train_gat2vec_bip(args.data, args.num_walks, args.walk_length, args.dimension,
                                 args.window_size, args.output)

    ''' for blogcatalog set multilabel = True'''
    if args.data == 'blogcatalog':
        multilabel = True
    else:
        multilabel = False

    c_eval = Classification(args.data, multilabel)
    result_df = c_eval.evaluate(model, args.label)
    print "Results ....."
    print result_df
    # g2v.param_walklen_nwalks('joint', args.data)
