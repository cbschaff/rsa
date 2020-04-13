import argparse
import dl


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train Agent.')
    parser.add_argument('--expdir', type=str, help='expdir', required=True)
    parser.add_argument('--gin_config', type=str, help='gin config', required=True)
    parser.add_argument('-b', '--gin_bindings', nargs='+', help='gin bindings to overwrite config')
    args = parser.parse_args()
    dl.load_config(args.gin_config, args.gin_bindings)
    dl.train(args.expdir)
