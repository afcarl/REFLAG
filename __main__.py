from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from src.Reflag import REFLAG
from Evaluation.Classification import Classification


def main():
    parser = ArgumentParser("REFLAG",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--data', nargs='?', required=True,
                        help='Input data directory in ../data/ folder')

    parser.add_argument('--label', nargs='?', default=False, type=bool,
                        help=' If data is labelled')

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
    reflag = REFLAG(args.data)
    model = reflag.train_REFLAG(args.data, args.label, args.num_walks, args.walk_length, args.dimension,
                                args.window_size, args.output)

    ''' for blogcatalog set multilabel = True'''
    c_eval = Classification(args.data, multilabel=False)
    c_eval.evaluate(model, args.label)