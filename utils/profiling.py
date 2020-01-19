import pstats
p = pstats.Stats("run_summarisation.cprof")
p.strip_dirs().sort_stats('cumulative').print_stats(20)
