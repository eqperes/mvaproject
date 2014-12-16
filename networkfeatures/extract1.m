net = load('../matconvnet-1.0-beta7/matlab/imagenet-vgg-f.mat');
list_names = {'Flickr_8k.trainImages', 'Flickr_8k.devImages', 'Flickr_8k.testImages'};
[~, list_size] = size(list_names);
for name_index=1:list_size
    for layer=16:22
        name = list_names{name_index};
        file_name = strcat(name, '.txt');
        mat_name = strcat(name, '.mat');
        names      = textread(strcat('../../Flickr8k_text/', file_name),'%s');
        features = networkfeatures(names, net, layer);
        save(strcat(['./layer' int2str(layer) '/'], mat_name),'features', 'names');
    end
end