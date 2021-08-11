import argparse
import json
import os
from main import train, eval, predict


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sound detection')

    argparser.add_argument('action', type=str,
                           help='what action to take (train, test, predict)')

    argparser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')

    argparser.add_argument('-f', '--file', default=None, type=str,
                           help='audio filename to prediction')
    argparser.add_argument('--model-path', default=None, type=str,
                           help='path to saved trained model file')
    argparser.add_argument('--model-cfg', default=None, type=str,
                           help='nn layer config file')

    args = argparser.parse_args()

    # Resolve config vs. resume
    checkpoint = None
    if args.config:
        config = json.load(open(args.config))
        config['net_mode'] = args.net_mode
        config['cfg'] = args.cfg

    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    # Pick mode to run
    if args.action == 'train':
        train(config)

    elif args.action == 'eval':
        eval(config)

    elif args.action == 'predict':
        predict(cfg)
