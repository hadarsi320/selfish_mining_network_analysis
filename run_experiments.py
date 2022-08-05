from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm

from main import mining_simulation
from utils import convert_args_dict, POWER_RANGE, args_to_outf


def main():
    default_args = {'num-nodes': 1000, 'turns': 1000, 'num-pools': 2, 'prints': 'parallel'}
    default_flags = []
    args_lists = []
    for tie_breaking in ['first', 'random']:
        for selfish_mining in [False, True]:
            for size in [0.1, 0.5, 0.9]:
                for connectivity in [0.1, 0.5, 0.9]:
                    for power in POWER_RANGE:
                        for seed in range(5):
                            args = default_args.copy()
                            flags = default_flags.copy()

                            if selfish_mining:
                                flags.append('selfish-mining')
                            args['tie-breaking'] = tie_breaking
                            args['pool-sizes'] = size
                            args['pool-connectivity'] = connectivity
                            args['pool-powers'] = power
                            args['seed'] = seed
                            args['outf'] = args_to_outf(args, flags)
                            if args['outf'].exists():
                                print('Skipping experiment as outf exists', args['outf'])
                                continue

                            cmd_args = convert_args_dict(args, flags)
                            args_lists.append(cmd_args)

    print('There are', len(args_lists), 'runs in total')
    start = datetime.now()
    with Pool() as p:
        for _ in tqdm(p.imap_unordered(mining_simulation, args_lists), total=len(args_lists)):
            pass

    print('The total runtime is ', datetime.now() - start)


if __name__ == '__main__':
    main()
