% For the common project with PGM

function [features] = extract_features(names)
	names_size = size(names);
	features = cell(names_size);
	for i=1:names_size
            name = names{i};
            im = imread(['../../../../Probabilistic_models/Project/Flicker8k_Dataset/' name]);
            im_ = single(im);
            %im_ = imresize(im_, net.normalization.imageSize(1:2));
            %im_ = im_ - net.normalization.averageImage;
            for o=1:2
            [~,feat] = getFeatures(im_, 'affineAdaptation',o==2) ;
            end
            features{i}=feat;
            
	end
end