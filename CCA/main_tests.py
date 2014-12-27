from performance_tests import PerfTests

view2path = '../captions/image_features_lda_200_topics.dict'
view1train_path = '../../../layers /layer21/Flickr_8k.trainImages.mat'
view1test_path = '../../../layers /layer21/Flickr_8k.testImages.mat'

perf = PerfTests(view1train_path, view1test_path, view2path)

