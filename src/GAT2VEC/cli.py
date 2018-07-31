from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from GAT2VEC.evaluation.classification import Classification
from GAT2VEC.evaluation.param_evaluation import param_walklen_nwalks
from GAT2VEC.gat2vec import Gat2Vec


def main():
    parser = ArgumentParser("gat2vec",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', nargs='?', required=True,
                        help='Input data directory')

    parser.add_argument('--output-dir', nargs='?', required=True,
                        help='Output directory')

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

    parser.add_argument('--multilabel', nargs='?', default=False, type=bool,
                        help='True if one node has multiple labels')

    args = parser.parse_args()
    TR = [0.1, 0.3, 0.5]  # the training ratio for classifier
    model = build_gat2vec_model(TR, args)
    evaluate_gat2vec_model(TR, args, model)
    # param_walklen_nwalks('joint', args.data, args.output_dir, TR, is_multilabel=args.multilabel)


def build_gat2vec_model(TR, args):
    g2v = Gat2Vec(args.data, args.output_dir, args.label, tr=TR)
    if args.algo == 'g2v':
        model = g2v.train_gat2vec(args.num_walks, args.walk_length, args.dimension,
                                  args.window_size, args.output)
    else:
        model = g2v.train_gat2vec_bip(args.num_walks, args.walk_length, args.dimension,
                                      args.window_size, args.output)
    return model


def evaluate_gat2vec_model(TR, args, model):
    c_eval = Classification(args.data, args.output_dir, TR, args.multilabel)
    result_df = c_eval.evaluate(model, args.label, evaluation_scheme="tr")
    print("Results .....")
    print(result_df)
