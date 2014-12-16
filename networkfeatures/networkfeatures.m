function [features] = networkfeatures(names, net, layer)
	names_size = size(names);
	features = cell(names_size);
	for i=1:names_size
		name = names{i};
		im = imread(['../../Flicker8k_Dataset/' name]);
		im_ = single(im);
		im_ = imresize(im_, net.normalization.imageSize(1:2));
		im_ = im_ - net.normalization.averageImage;
		res = vl_simplenn(net, im_);
		features{i} = res(layer).x;
	end
end