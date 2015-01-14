% For the common project with PGM

setup ;

% build the visual vocabulary

% list = textread('../../../../Probabilistic_models/Project/Flickr8k_text/Flickr_8k.trainImages.txt','%s');
% dev_data = textread('../../../../Probabilistic_models/Project/Flickr8k_text/Flickr_8k.devImages.txt','%s');
% test_data=textread('../../../../Probabilistic_models/Project/Flickr8k_text/Flickr_8k.testImages.txt','%s');
% %list=cat(1,list,test_data);
% %list=cat(1,list,test_dev);
% vocabulary = computeVocabularyFromImageList2(list);
% save('../../../../Probabilistic_models/Project/SIFT_data/vocabulary.mat','vocabulary');
% vocabulary_dev = computeVocabularyFromImageList2(dev_data);
% save('../../../../Probabilistic_models/Project/SIFT_data/vocabulary_dev.mat','vocabulary_dev');
% vocabulary_test = computeVocabularyFromImageList2(test_data);
% save('../../../../Probabilistic_models/Project/SIFT_data/vocabulary_test.mat','vocabulary_test');

% compute the spatial histograms

vocabulary = load('../../../../Probabilistic_models/Project/SIFT_data/Flickr_SIFT_train.mat');
names      = textread('../../../../Probabilistic_models/Project/Flickr8k_text/Flickr_8k.trainImages.txt','%s')
features = computeHistogramsFromImageList2(vocabulary, names);
save('../../../../Probabilistic_models/Project/SIFT_data/histograms/Flickr8k_train2.mat','features','names');

%vocabulary_dev = load('../../../../Probabilistic_models/Project/SIFT_data/vocabulary_dev.mat');
names      = textread('../../../../Probabilistic_models/Project/Flickr8k_text/Flickr_8k.devImages.txt','%s')
features = computeHistogramsFromImageList2(vocabulary, names);
save('../../../../Probabilistic_models/Project/SIFT_data/histograms/Flickr8k_dev2.mat','features','names');

%vocabulary_test = load('../../../../Probabilistic_models/Project/SIFT_data/vocabulary_test.mat');
names      = textread('../../../../Probabilistic_models/Project/Flickr8k_text/Flickr_8k.testImages.txt','%s')
features = computeHistogramsFromImageList2(vocabulary, names);
save('../../../../Probabilistic_models/Project/SIFT_data/histograms/Flickr8k_test2.mat','features','names');

% % visualize computed features for one image
im                       = imread('../../../../Probabilistic_models/Project/Flicker8k_Dataset/667626_18933d713e.jpg');
[keypoints, descriptors] = computeFeatures(im);
col     = keypoints(1,:);
row     = keypoints(2,:);
binsize = keypoints(4,:); % bin size sift descriptor (pixels).
                           
% recall that sift is composed of a spatial grid of 4x4 bins
radius  = binsize*2;      % visualize sift by a circle with radius of 2 bin widths.
 
figure(1); clf; imagesc(im);
vl_plotframe([col(1:50:end); row(1:50:end); radius(1:50:end)]); % visual keypoints as circles with radius scale
axis image;





