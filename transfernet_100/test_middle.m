%清除command window and workspace
clear all;
close all;
clc;

flag=0;  %用于观察训练到第几个文件夹
rows=256;  %压缩后图像大小
classes=30;   %该值表示当前要处理的类别的个数，因为数据类别数过大会导致out of memory
dir_name='svg-5strokes-中笔画(15-18笔)';        %%%%%%%
totalNum=classes*20;

data=zeros(rows,rows,totalNum);   %%data用于存储压缩过后的图像数据
labels=zeros(1,totalNum);   %labels存储图像对应的类别
% set=ones(1,totalNum);   %set区别训练集和测试集 set=1表示训练集 set=3表示测试集

fileFolder=fullfile('E:\刘杨',dir_name);
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';    %fileNames存储的是所有30个类的类名

numOfPic=0;  %用于记录压缩到第几张图像

%依次对各个类别的图片进行处理
for filenum= 3:32    
    files=dir(fullfile(fileFolder,dirOutput(filenum).name,'*_4.png'));   %%%%%%%
    fileNames2={files.name}';   %fileNames2存储的是当前类的所有后缀为*_1.png的文件的名字
    num=size(files,1);
%     trainNum=num*3/4;
%     testNum=num/4;
    maindir=[fileFolder,'\',dirOutput(filenum).name];
    
    for i=61:num
        numOfPic=numOfPic+1;
        path=[maindir,'\',files(i).name];
        f=imread(path);
        im=imresize(f(:,:,1),[rows rows],'bilinear');  %降维到256*256
        data(:,:,numOfPic)=im;        
    end    
    flag=flag+1 
end
labels(1:20) = 7;
labels(21:40) = 10;
labels(41:60) = 23;
labels(61:80) = 36;
labels(81:100) = 41;
labels(101:120) = 42;
labels(121:140) = 44;
labels(141:160) = 59;
labels(161:180) = 64;
labels(181:200) = 66;
labels(201:220) = 67;
labels(221:240) = 75;
labels(241:260) = 77;
labels(261:280) = 79;
labels(281:300) = 96;
labels(301:320) = 114;
labels(321:340) = 136;
labels(341:360) = 146;
labels(361:380) = 147;
labels(381:400) = 167;
labels(401:420) = 171;
labels(421:440) = 179;
labels(441:460) = 200;
labels(461:480) = 223;
labels(481:500) = 229;
labels(501:520) = 234;
labels(521:540) = 238;
labels(541:560) = 242;
labels(561:580) = 244;
labels(581:600) = 245;

%将data，labels和set合并存储为imdb    
imdb.images.data = data;
imdb.images.labels = labels;
 
%存储imdb为mat格式
dir='E:\刘杨';
savefile=[dir,'\','Data','\',num2str(rows),'_100_middle','.mat'];          %%%%%%
save(savefile,'imdb','-v7.3');