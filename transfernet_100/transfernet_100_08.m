function [net, info] = transfernet_100_08(varargin)

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..','matconvnet-1.0-beta7', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile('E:\ÁõÑî\transfernet_100\data','sketch') ;
opts.expDir = fullfile('E:\ÁõÑî\transfernet_100\data','sketch-baseline') ;
opts.imdbPath = fullfile(opts.dataDir);
opts.train.batchSize = 100 ;
opts.train.numEpochs = 35 ;
opts.train.continue = true ;
opts.train.useGpu = false ;
opts.train.learningRate = [0.001*ones(1, 12) 0.0001*ones(1,6) 0.00001] ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

load('E:\ÁõÑî\transfernet_100\data\sketch-baseline-Ö»µ÷Õûfc²ã 0.01 0.01 4096 40%\net-epoch-19.mat');

imdb1 = load(fullfile(opts.imdbPath,'256_40_01.mat')) ;
imageSize = 227 ;
imdb1.imdb.images.Data = zeros(imageSize,imageSize,20000);
imdb1.imdb.images.Set = zeros(1,20000);

for xx =1:4000
    for yy =1:5
        imdb1.imdb.images.Set(:,(xx-1)*5+yy) = imdb1.imdb.images.set(:,xx);
    end
end

imdb1.imdb.images.Labels = zeros(1,20000);
for xx =1:4000
    for yy =1:5
        imdb1.imdb.images.Labels(:,(xx-1)*5+yy) = imdb1.imdb.images.labels(:,xx);
    end
end

for i = 1:4000
    im1 = imdb1.imdb.images.data(1:227,1:227,i);
    im2 = imdb1.imdb.images.data(1:227,30:256,i);
    im3 = imdb1.imdb.images.data(30:256,1:227,i);
    im4 = imdb1.imdb.images.data(30:256,30:256,i);
    im5 = imdb1.imdb.images.data(15:241,15:241,i);
  
    imdb1.imdb.images.Data(:,:,(i-1)*5+1) = im1 ;
    imdb1.imdb.images.Data(:,:,(i-1)*5+2) = im2 ;
    imdb1.imdb.images.Data(:,:,(i-1)*5+3) = im3 ;
    imdb1.imdb.images.Data(:,:,(i-1)*5+4) = im4 ;
    imdb1.imdb.images.Data(:,:,(i-1)*5+5) = im5 ;
end

imdb2 = load(fullfile(opts.imdbPath,'256_40_02.mat')) ;
imdb2.imdb.images.Data = zeros(imageSize,imageSize,20000);
imdb2.imdb.images.Set = zeros(1,20000);

for xx =1:4000
    for yy =1:5
        imdb2.imdb.images.Set(:,(xx-1)*5+yy) = imdb2.imdb.images.set(:,xx);
    end
end

imdb2.imdb.images.Labels = zeros(1,20000);
for xx =1:4000
    for yy =1:5
        imdb2.imdb.images.Labels(:,(xx-1)*5+yy) = imdb2.imdb.images.labels(:,xx);
    end
end

for i = 1:4000
    im1 = imdb2.imdb.images.data(1:227,1:227,i);
    im2 = imdb2.imdb.images.data(1:227,30:256,i);
    im3 = imdb2.imdb.images.data(30:256,1:227,i);
    im4 = imdb2.imdb.images.data(30:256,30:256,i);
    im5 = imdb2.imdb.images.data(15:241,15:241,i);
  
    imdb2.imdb.images.Data(:,:,(i-1)*5+1) = im1 ;
    imdb2.imdb.images.Data(:,:,(i-1)*5+2) = im2 ;
    imdb2.imdb.images.Data(:,:,(i-1)*5+3) = im3 ;
    imdb2.imdb.images.Data(:,:,(i-1)*5+4) = im4 ;
    imdb2.imdb.images.Data(:,:,(i-1)*5+5) = im5 ;
end

imdb3 = load(fullfile(opts.imdbPath,'256_40_03.mat')) ;
imdb3.imdb.images.Data = zeros(imageSize,imageSize,20000);
imdb3.imdb.images.Set = zeros(1,20000);

for xx =1:4000
    for yy =1:5
        imdb3.imdb.images.Set(:,(xx-1)*5+yy) = imdb3.imdb.images.set(:,xx);
    end
end

imdb3.imdb.images.Labels = zeros(1,20000);
for xx =1:4000
    for yy =1:5
        imdb3.imdb.images.Labels(:,(xx-1)*5+yy) = imdb3.imdb.images.labels(:,xx);
    end
end

for i = 1:4000
    im1 = imdb3.imdb.images.data(1:227,1:227,i);
    im2 = imdb3.imdb.images.data(1:227,30:256,i);
    im3 = imdb3.imdb.images.data(30:256,1:227,i);
    im4 = imdb3.imdb.images.data(30:256,30:256,i);
    im5 = imdb3.imdb.images.data(15:241,15:241,i);
  
    imdb3.imdb.images.Data(:,:,(i-1)*5+1) = im1 ;
    imdb3.imdb.images.Data(:,:,(i-1)*5+2) = im2 ;
    imdb3.imdb.images.Data(:,:,(i-1)*5+3) = im3 ;
    imdb3.imdb.images.Data(:,:,(i-1)*5+4) = im4 ;
    imdb3.imdb.images.Data(:,:,(i-1)*5+5) = im5 ;
end

imdb4 = load(fullfile(opts.imdbPath,'256_40_04.mat')) ;
imdb4.imdb.images.Data = zeros(imageSize,imageSize,20000);
imdb4.imdb.images.Set = zeros(1,20000);

for xx =1:4000
    for yy =1:5
        imdb4.imdb.images.Set(:,(xx-1)*5+yy) = imdb4.imdb.images.set(:,xx);
    end
end

imdb4.imdb.images.Labels = zeros(1,20000);
for xx =1:4000
    for yy =1:5
        imdb4.imdb.images.Labels(:,(xx-1)*5+yy) = imdb4.imdb.images.labels(:,xx);
    end
end

for i = 1:4000
    im1 = imdb4.imdb.images.data(1:227,1:227,i);
    im2 = imdb4.imdb.images.data(1:227,30:256,i);
    im3 = imdb4.imdb.images.data(30:256,1:227,i);
    im4 = imdb4.imdb.images.data(30:256,30:256,i);
    im5 = imdb4.imdb.images.data(15:241,15:241,i);
 
    imdb4.imdb.images.Data(:,:,(i-1)*5+1) = im1 ;
    imdb4.imdb.images.Data(:,:,(i-1)*5+2) = im2 ;
    imdb4.imdb.images.Data(:,:,(i-1)*5+3) = im3 ;
    imdb4.imdb.images.Data(:,:,(i-1)*5+4) = im4 ;
    imdb4.imdb.images.Data(:,:,(i-1)*5+5) = im5 ;
end

imdb5 = load(fullfile(opts.imdbPath,'256_40_05.mat')) ;
imdb5.imdb.images.Data = zeros(imageSize,imageSize,20000);
imdb5.imdb.images.Set = zeros(1,20000);

for xx =1:4000
    for yy =1:5
        imdb5.imdb.images.Set(:,(xx-1)*5+yy) = imdb5.imdb.images.set(:,xx);
    end
end

imdb5.imdb.images.Labels = zeros(1,20000);
for xx =1:4000
    for yy =1:5
        imdb5.imdb.images.Labels(:,(xx-1)*5+yy) = imdb5.imdb.images.labels(:,xx);
    end
end

for i = 1:4000
    im1 = imdb5.imdb.images.data(1:227,1:227,i);
    im2 = imdb5.imdb.images.data(1:227,30:256,i);
    im3 = imdb5.imdb.images.data(30:256,1:227,i);
    im4 = imdb5.imdb.images.data(30:256,30:256,i);
    im5 = imdb5.imdb.images.data(15:241,15:241,i);
  
    imdb5.imdb.images.Data(:,:,(i-1)*5+1) = im1 ;
    imdb5.imdb.images.Data(:,:,(i-1)*5+2) = im2 ;
    imdb5.imdb.images.Data(:,:,(i-1)*5+3) = im3 ;
    imdb5.imdb.images.Data(:,:,(i-1)*5+4) = im4 ;
    imdb5.imdb.images.Data(:,:,(i-1)*5+5) = im5 ;
end

imdb1.imdb.images.Data = 255 - imdb1.imdb.images.Data;
imdb2.imdb.images.Data = 255 - imdb2.imdb.images.Data;
imdb3.imdb.images.Data = 255 - imdb3.imdb.images.Data;
imdb4.imdb.images.Data = 255 - imdb4.imdb.images.Data;
imdb5.imdb.images.Data = 255 - imdb5.imdb.images.Data;

[net, info] = cnn_train_100_08(net, imdb1, imdb2, imdb3, imdb4, imdb5, @getBatch, ...
    opts.train) ;

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb1, imdb2, imdb3, imdb4, imdb5, batch)
% --------------------------------------------------------------------
for x=1:numel(batch)
    if batch(x)>=1 && batch(x)<=20000
        im(:,:,1,x) = imdb1.imdb.images.Data(:,:,batch(x));
    elseif batch(x)>=20001 && batch(x)<=40000
        im(:,:,1,x) = imdb2.imdb.images.Data(:,:,batch(x)-20000);
    elseif batch(x)>=40001 && batch(x)<=60000
        im(:,:,1,x) = imdb3.imdb.images.Data(:,:,batch(x)-40000);
    elseif batch(x)>=60001 && batch(x)<=80000
        im(:,:,1,x) = imdb4.imdb.images.Data(:,:,batch(x)-60000);
    else
        im(:,:,1,x) = imdb5.imdb.images.Data(:,:,batch(x)-80000);
    end
end

im = repmat(im, [1 1 3]);

for x=1:numel(batch)
    if batch(x)>=1 && batch(x)<=20000
        labels(:,x) = imdb1.imdb.images.Labels(:,batch(x));
    elseif batch(x)>=20001 && batch(x)<=40000
        labels(:,x) = imdb2.imdb.images.Labels(:,batch(x)-20000);
    elseif batch(x)>=40001 && batch(x)<=60000
        labels(:,x) = imdb3.imdb.images.Labels(:,batch(x)-40000);
    elseif batch(x)>=60001 && batch(x)<=80000
        labels(:,x) = imdb4.imdb.images.Labels(:,batch(x)-60000);
    else
        labels(:,x) = imdb5.imdb.images.Labels(:,batch(x)-80000);
    end
end