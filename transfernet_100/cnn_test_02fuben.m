clear all;close all;clc;
run('E:\刘杨\matconvnet-1.0-beta7\matlab\vl_setupnn.m');

opts.dataDir = fullfile('E:\刘杨\transfernet_100\data','sketch') ;
opts.imdbPath = fullfile(opts.dataDir);
imdb1 = load(fullfile(opts.imdbPath,'256_80_01.mat')) ;    %%%%%%%%
imdb2 = load(fullfile(opts.imdbPath,'256_80_02.mat')) ;
imdb3 = load(fullfile(opts.imdbPath,'256_80_03.mat')) ;
imdb4 = load(fullfile(opts.imdbPath,'256_80_04.mat')) ;
imdb5 = load(fullfile(opts.imdbPath,'256_80_05.mat')) ;
%%%%%%%%%%
net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 单倍数据 40% conv5\net-epoch-20.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
   
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','单倍80送入40模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5


net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 单倍数据 60% conv5 conv4\net-epoch-20.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
   
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','单倍80送入60模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5


net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 单倍数据 80% conv5 conv4 conv3\net-epoch-20.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
   
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','单倍80送入80模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 单倍数据\net-epoch-22.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)

    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','单倍80送入100模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 五倍数据 40% conv5\net-epoch-13.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
  
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','五倍80送入40模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 五倍数据 60% conv5 conv4\net-epoch-13.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
 
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','五倍80送入60模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 五倍数据 80% conv5 conv4 conv3\net-epoch-8.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
  
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','五倍80送入80模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%  


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-一轮调整 0.01 0.01 4096 五倍数据\net-epoch-11.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
  
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','五倍80送入100模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-调整两次 0.01 0.01 4096 第一次1倍数据 第二次5倍数据 40%\net-epoch-18.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)

    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','两次80送入40模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-调整两次 0.01 0.01 4096 第一次1倍数据 第二次5倍数据 60%\net-epoch-18.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
 
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','两次80送入60模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-调整两次 0.01 0.01 4096 第一次1倍数据 第二次5倍数据 80%\net-epoch-19.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)
  
    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','两次80送入80模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\刘杨\transfernet_100\data\sketch-baseline-调整两次 0.01 0.01 4096 第一次1倍数据 第二次5倍数据\net-epoch-20.mat');

test1 = imdb1.imdb.images.data(:,:,imdb1.imdb.images.set==3) ;
test2 = imdb2.imdb.images.data(:,:,imdb2.imdb.images.set==3) ;
test3 = imdb3.imdb.images.data(:,:,imdb3.imdb.images.set==3) ;
test4 = imdb4.imdb.images.data(:,:,imdb4.imdb.images.set==3) ;
test5 = imdb5.imdb.images.data(:,:,imdb5.imdb.images.set==3) ;

testX1 = cat(3,test1,test2,test3,test4,test5) ;

testX1 = single(255-testX1) ;

testx1 = zeros(256,256,1,5000);

result = zeros(250,5000);    %%%%%

for i = 1:5000
    testx1(:,:,:,i) = testX1(:,:,i) ;
end
testx1 = repmat(testx1, [1 1 3]);
testx1 = single(testx1) ;

label1 = imdb1.imdb.images.labels(:,imdb1.imdb.images.set==3) ;
label2 = imdb2.imdb.images.labels(:,imdb2.imdb.images.set==3) ;
label3 = imdb3.imdb.images.labels(:,imdb3.imdb.images.set==3) ;
label4 = imdb4.imdb.images.labels(:,imdb4.imdb.images.set==3) ;
label5 = imdb5.imdb.images.labels(:,imdb5.imdb.images.set==3) ;
labels = cat(2,label1,label2,label3,label4,label5) ;

predY1 = zeros(1,length(labels)) ;

for i = 1:length(labels)

    testimg = testx1(:,:,:,i);
    %对图像进行crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %对图像进行翻转
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %把两种图像进行联合作为输入
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %使用训练模型进行计算各类值
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %对每张图都进行计算，得到每行10个值
    pred1 = squeeze(gather(res(end-1).x));
    %对10个值求和
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %找出250中的最大值
    [~,pred1] = max(pred1);
    %赋值给预测函数Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\刘杨';
savefile=[dir,'\','transfernet_100','\','两次80送入100模型','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%