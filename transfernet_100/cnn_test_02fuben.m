clear all;close all;clc;
run('E:\����\matconvnet-1.0-beta7\matlab\vl_setupnn.m');

opts.dataDir = fullfile('E:\����\transfernet_100\data','sketch') ;
opts.imdbPath = fullfile(opts.dataDir);
imdb1 = load(fullfile(opts.imdbPath,'256_80_01.mat')) ;    %%%%%%%%
imdb2 = load(fullfile(opts.imdbPath,'256_80_02.mat')) ;
imdb3 = load(fullfile(opts.imdbPath,'256_80_03.mat')) ;
imdb4 = load(fullfile(opts.imdbPath,'256_80_04.mat')) ;
imdb5 = load(fullfile(opts.imdbPath,'256_80_05.mat')) ;
%%%%%%%%%%
net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �������� 40% conv5\net-epoch-20.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����40ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5


net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �������� 60% conv5 conv4\net-epoch-20.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����60ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5


net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �������� 80% conv5 conv4 conv3\net-epoch-20.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����80ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 ��������\net-epoch-22.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����100ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �屶���� 40% conv5\net-epoch-13.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','�屶80����40ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �屶���� 60% conv5 conv4\net-epoch-13.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','�屶80����60ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �屶���� 80% conv5 conv4 conv3\net-epoch-8.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','�屶80����80ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%  


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-һ�ֵ��� 0.01 0.01 4096 �屶����\net-epoch-11.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','�屶80����100ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-�������� 0.01 0.01 4096 ��һ��1������ �ڶ���5������ 40%\net-epoch-18.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����40ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%

clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-�������� 0.01 0.01 4096 ��һ��1������ �ڶ���5������ 60%\net-epoch-18.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����60ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-�������� 0.01 0.01 4096 ��һ��1������ �ڶ���5������ 80%\net-epoch-19.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����80ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%


clearvars -except opts imdb1 imdb2 imdb3 imdb4 imdb5

net1 = load('E:\����\transfernet_100\data\sketch-baseline-�������� 0.01 0.01 4096 ��һ��1������ �ڶ���5������\net-epoch-20.mat');

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
    %��ͼ�����crop
    im1 = testimg(1:227,1:227,:) ;
    im2 = testimg(30:256,1:227,:) ;
    im3 = testimg(1:227,30:256,:) ;
    im4 = testimg(30:256,30:256,:) ;
    im5 = testimg(15:241,15:241,:) ;
    im15 = cat(4,im1,im2,im3,im4,im5);
    %��ͼ����з�ת
    im15_lr=cat(4,fliplr(testimg(1:227,1:227)),fliplr(testimg(30:256,1:227)),fliplr(testimg(1:227,30:256)),fliplr(testimg(30:256,30:256)),fliplr(testimg(15:241,15:241)));
    im15_lr = repmat(im15_lr, [1 1 3]);
    %������ͼ�����������Ϊ����
    im_new = cat(4,im15,im15_lr);
    
    net1.net.layers{end}.class = kron(ones(1,10),labels(i));
    %ʹ��ѵ��ģ�ͽ��м������ֵ
    res = vl_simplenn(net1.net,im_new,[],[],'disableDropout',true,'conserveMemory',false,'sync',true);
    %��ÿ��ͼ�����м��㣬�õ�ÿ��10��ֵ
    pred1 = squeeze(gather(res(end-1).x));
    %��10��ֵ���
%     pred = max(pred,[],2);
    
    pred1 = sum(pred1,2);

    result(:,i) = pred1;  %%%%%
    %�ҳ�250�е����ֵ
    [~,pred1] = max(pred1);
    %��ֵ��Ԥ�⺯��Y
    predY1(i) = pred1;      
end

mean(predY1==labels)

dir='E:\����';
savefile=[dir,'\','transfernet_100','\','����80����100ģ��','.mat'];     %%%%%%
save(savefile,'result','-v7.3');   %%%%%%%