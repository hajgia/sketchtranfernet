function [net, info] = transfernet_100_04(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','matconvnet-1.0-beta7', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('E:\刘杨\transfernet_100\data','sketch') ;
opts.expDir = fullfile('E:\刘杨\transfernet_100\data','sketch-baseline') ;
opts.imdbPath = fullfile(opts.dataDir);
opts.train.batchSize = 100 ;
opts.train.numEpochs = 50 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.001*ones(1, 12) 0.0001*ones(1,6) 0.00001] ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

load('E:\刘杨\transfernet_100\data\sketch-baseline-只调整fc 0.01 0.01 4096 5倍数据  三层fc\net-epoch-17.mat');
% Layers = {} ;
% %-----------------------------------------------
% Layers{1}.type = 'conv' ;
% Layers{1}.filters = params(1).value ;
% Layers{1}.biases = params(2).value ;
% Layers{1}.stride = 4 ;
% Layers{1}.pad = 0 ;
% Layers{1}.learningRate = [1,2] ;
% Layers{1}.weightDecay = [1,0] ;
% 
% Layers{2}.type = 'relu' ;
% 
% Layers{3}.type = 'pool' ;
% Layers{3}.method = 'max' ;
% Layers{3}.pool = [3,3] ;
% Layers{3}.stride = 2 ;
% Layers{3}.pad = 0 ;
% 
% Layers{4}.type = 'conv';
% Layers{4}.filters = params(3).value ;
% Layers{4}.biases = params(4).value ;
% Layers{4}.stride = 1 ;
% Layers{4}.pad = 2 ;
% Layers{4}.learningRate = [1,2] ;
% Layers{4}.weightDecay = [1,0] ;
% 
% Layers{5}.type = 'relu' ;
% 
% Layers{6}.type = 'pool' ;
% Layers{6}.method = 'max' ;
% Layers{6}.pool = [3,3] ;
% Layers{6}.stride = 2 ;
% Layers{6}.pad = 0 ;
% 
% Layers{7}.type = 'conv';
% Layers{7}.filters = params(5).value ;
% Layers{7}.biases = params(6).value ;
% Layers{7}.stride = 1 ;
% Layers{7}.pad = 1 ;
% Layers{7}.learningRate = [1,2] ;
% Layers{7}.weightDecay = [1,0] ;
% 
% Layers{8}.type = 'relu' ;
% 
% Layers{9}.type = 'conv';
% Layers{9}.filters = params(7).value ;
% Layers{9}.biases = params(8).value ;
% Layers{9}.stride = 1 ;
% Layers{9}.pad = 1 ;
% Layers{9}.learningRate = [1,2] ;
% Layers{9}.weightDecay = [1,0] ;
% 
% Layers{10}.type = 'relu' ;
% 
% Layers{11}.type = 'conv';
% Layers{11}.filters = params(9).value ;
% Layers{11}.biases = params(10).value ;
% Layers{11}.stride = 1 ;
% Layers{11}.pad = 1 ;
% Layers{11}.learningRate = [1,2] ;
% Layers{11}.weightDecay = [1,0] ;
% 
% Layers{12}.type = 'relu' ;
% 
% Layers{13}.type = 'pool' ;
% Layers{13}.method = 'max' ;
% Layers{13}.pool = [3,3] ;
% Layers{13}.stride = 2 ;
% Layers{13}.pad = 0 ;
% 
% Layers{14}.type = 'conv';
% Layers{14}.filters = params(11).value ;
% Layers{14}.biases = params(12).value ;
% Layers{14}.stride = 1 ;
% Layers{14}.pad = 0 ;
% Layers{14}.learningRate = [1,2] ;
% Layers{14}.weightDecay = [1,0] ;
% 
% Layers{15}.type = 'relu' ;
%-----------------------------------------------
% imdb1 = load(fullfile(opts.imdbPath,'256_100_01.mat')) ;
% imageSize = 227 ;
% imdb1.imdb.images.Data = zeros(imageSize,imageSize,4000);
% for i = 1:4000
%     im = imresize(imdb1.imdb.images.data(:,:,i),[imageSize,imageSize],'bilinear') ;
%     imdb1.imdb.images.Data(:,:,i) = im ;
% end
% imdb2 = load(fullfile(opts.imdbPath,'256_100_02.mat')) ;
% imdb2.imdb.images.Data = zeros(imageSize,imageSize,4000);
% for i = 1:4000
%     im = imresize(imdb2.imdb.images.data(:,:,i),[imageSize,imageSize],'bilinear') ;
%     imdb2.imdb.images.Data(:,:,i) = im ;
% end
% imdb3 = load(fullfile(opts.imdbPath,'256_100_03.mat')) ;
% imdb3.imdb.images.Data = zeros(imageSize,imageSize,4000);
% for i = 1:4000
%     im = imresize(imdb3.imdb.images.data(:,:,i),[imageSize,imageSize],'bilinear') ;
%     imdb3.imdb.images.Data(:,:,i) = im ;
% end
% imdb4 = load(fullfile(opts.imdbPath,'256_100_04.mat')) ;
% imdb4.imdb.images.Data = zeros(imageSize,imageSize,4000);
% for i = 1:4000
%     im = imresize(imdb4.imdb.images.data(:,:,i),[imageSize,imageSize],'bilinear') ;
%     imdb4.imdb.images.Data(:,:,i) = im ;
% end
% imdb5 = load(fullfile(opts.imdbPath,'256_100_05.mat')) ;
% imdb5.imdb.images.Data = zeros(imageSize,imageSize,4000);
% for i = 1:4000
%     im = imresize(imdb5.imdb.images.data(:,:,i),[imageSize,imageSize],'bilinear') ;
%     imdb5.imdb.images.Data(:,:,i) = im ;
% end


imdb1 = load(fullfile(opts.imdbPath,'256_100_01.mat')) ;
imageSize = 227 ;
imdb1.imdb.images.Data = zeros(imageSize,imageSize,4000);
for i = 1:4000
    im = imdb1.imdb.images.data(15:241,15:241,i);
    imdb1.imdb.images.Data(:,:,i) = im ;
end
imdb2 = load(fullfile(opts.imdbPath,'256_100_02.mat')) ;
imdb2.imdb.images.Data = zeros(imageSize,imageSize,4000);
for i = 1:4000
    im = imdb2.imdb.images.data(15:241,15:241,i);
    imdb2.imdb.images.Data(:,:,i) = im ;
end
imdb3 = load(fullfile(opts.imdbPath,'256_100_03.mat')) ;
imdb3.imdb.images.Data = zeros(imageSize,imageSize,4000);
for i = 1:4000
    im = imdb3.imdb.images.data(15:241,15:241,i);
    imdb3.imdb.images.Data(:,:,i) = im ;
end
imdb4 = load(fullfile(opts.imdbPath,'256_100_04.mat')) ;
imdb4.imdb.images.Data = zeros(imageSize,imageSize,4000);
for i = 1:4000
    im = imdb4.imdb.images.data(15:241,15:241,i);
    imdb4.imdb.images.Data(:,:,i) = im ;
end
imdb5 = load(fullfile(opts.imdbPath,'256_100_05.mat')) ;
imdb5.imdb.images.Data = zeros(imageSize,imageSize,4000);
for i = 1:4000
    im = imdb5.imdb.images.data(15:241,15:241,i);
    imdb5.imdb.images.Data(:,:,i) = im ;
end

% net.layers = Layers(1:15);
% net.layers{end+1} = struct('type', 'conv', ...                  %16
%                            'filters', 0.01 * randn(1,1,4096,2048,'single'),...
%                            'biases', 0.01*ones(1,2048,'single'), ...
%                            'stride', 1, ...
%                            'pad', 0, ...
%                            'filtersLearningRate', 1, ...
%                            'biasesLearningRate', 2, ...
%                            'filtersWeightDecay', 1, ...
%                            'biasesWeightDecay', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;     %17
% net.layers{end+1} = struct('type', 'dropout', ...    %18
%                            'rate', 0.5) ;
% net.layers{end+1} = struct('type', 'conv', ...       %19
%                            'filters', 0.01 * randn(1,1,2048,250,'single'),...
%                            'biases', 0.01*ones(1,250,'single'), ...
%                            'stride', 1, ...
%                            'pad', 0, ...
%                            'filtersLearningRate', 1, ...
%                            'biasesLearningRate', 2, ...
%                            'filtersWeightDecay', 1, ...
%                            'biasesWeightDecay', 0) ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;     %20

imdb1.imdb.images.Data = 255 - imdb1.imdb.images.Data;
imdb2.imdb.images.Data = 255 - imdb2.imdb.images.Data;
imdb3.imdb.images.Data = 255 - imdb3.imdb.images.Data;
imdb4.imdb.images.Data = 255 - imdb4.imdb.images.Data;
imdb5.imdb.images.Data = 255 - imdb5.imdb.images.Data;

[net, info] = cnn_train_100_04(net, imdb1, imdb2, imdb3, imdb4, imdb5, @getBatch, ...
    opts.train) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb1, imdb2, imdb3, imdb4, imdb5, batch)
% --------------------------------------------------------------------
for x=1:numel(batch)
    if batch(x)>=1 && batch(x)<=4000
        im(:,:,1,x) = imdb1.imdb.images.Data(:,:,batch(x));
    elseif batch(x)>=4001 && batch(x)<=8000
        im(:,:,1,x) = imdb2.imdb.images.Data(:,:,batch(x)-4000);
    elseif batch(x)>=8001 && batch(x)<=12000
        im(:,:,1,x) = imdb3.imdb.images.Data(:,:,batch(x)-8000);
    elseif batch(x)>=12001 && batch(x)<=16000
        im(:,:,1,x) = imdb4.imdb.images.Data(:,:,batch(x)-12000);
    else
        im(:,:,1,x) = imdb5.imdb.images.Data(:,:,batch(x)-16000);
    end
end

im = repmat(im, [1 1 3]);

for x=1:numel(batch)
    if batch(x)>=1 && batch(x)<=4000
        labels(:,x) = imdb1.imdb.images.labels(:,batch(x));
    elseif batch(x)>=4001 && batch(x)<=8000
        labels(:,x) = imdb2.imdb.images.labels(:,batch(x)-4000);
    elseif batch(x)>=8001 && batch(x)<=12000
        labels(:,x) = imdb3.imdb.images.labels(:,batch(x)-8000);
    elseif batch(x)>=12001 && batch(x)<=16000
        labels(:,x) = imdb4.imdb.images.labels(:,batch(x)-12000);
    else
        labels(:,x) = imdb5.imdb.images.labels(:,batch(x)-16000);
    end
end
            


