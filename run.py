import argparse
import find_patterns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Sound detection')

    argparser.add_argument('-p', '--pattern', type=str,
                           help="Audio-pattern, which need to be detected in audio-stream")

    argparser.add_argument('-a', '--audio', type=str,
                           help="Audio-stream, in which audio-pattern need to be detected ")

    argparser.add_argument('--sr', default=22050, type=int,
                           help="Sample rate (default 22050)")

    argparser.add_argument('--n-mfcc', default=20, type=int,
                           help="Number of MFCCs")

    argparser.add_argument('-t', '--threshold', default=0.8, type=float,
                           help="Minimal degree of similarity between the pattern and the audio-stream window"
                                "(between 0 and 1)")

    argparser.add_argument('-q', '--cut-quantile', default=0.96, type=float,
                           help="Cut pattern parameter (between 0 and 1)")

    args = argparser.parse_args()

    if args.pattern is None or args.audio is None:
        argparser.error("Arguments 'pattern' and 'audio' must be specified")

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    fig.tight_layout(pad=2.0)
    find_patterns.find_patterns(args.pattern, args.audio, sr=args.sr, n_mfcc=args.n_mfcc,
                                threshold=args.threshold, q=args.cut_quantile)
    plt.show()
