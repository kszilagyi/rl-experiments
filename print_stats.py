import pstats
from pstats import SortKey

if __name__ == '__main__':
    p = pstats.Stats('main_stats')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(30)

