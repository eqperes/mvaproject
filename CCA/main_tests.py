import sys
from performance_tests import PerfTests

view2path = sys.argv[1]
view1train_path = '../../layers /layer20/Flickr_8k.trainImages.mat'
view1test_path = '../../layers /layer20/Flickr_8k.testImages.mat'

perf = PerfTests(view1train_path, view1test_path, view2path)
perf.CCAfy(1)

for i in [1, 5, 10, 50, 100, 200, 500, 1000]:
	print perf.score_test(i, False, normalize=False)
	print perf.rank_test(i, False, normalize=False)