import argparse
import pickle
import pprint

import pandas as pd


def main(trials_file):
    trials = pickle.load(open(trials_file, 'rb'))
    df = pd.DataFrame(trials.vals)
    df['score_tr'] = [r['score_tr'] for r in trials.results]
    df['score_val'] = [r['score_val'] for r in trials.results]
    df['book_time'] = [t['book_time'].strftime('%Y-%m-%d %H:%M:%S') for t in trials.trials]
    pd.set_option('expand_frame_repr', False)
    print("All trials")
    print(df)
    print("Best 10 trials")
    print(df.sort_values(by=['score_val']).head(10))

    vals = trials.best_trial['misc']['vals']
    vals = {k: v[0] for k, v in vals.items()}
    pprint.pprint(vals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--trials_file',
        type=str,
        default='artifacts/stg0/1_cnn4l_aux_trials.pickle'
    )
    args, _ = parser.parse_known_args()

    main(args.trials_file)
