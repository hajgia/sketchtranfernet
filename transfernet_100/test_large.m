%���command window and workspace
clear all;
close all;
clc;

flag=0;  %���ڹ۲�ѵ�����ڼ����ļ���
rows=256;  %ѹ����ͼ���С
classes=30;   %��ֵ��ʾ��ǰҪ��������ĸ�������Ϊ�������������ᵼ��out of memory
dir_name='svg-5strokes-��ʻ�(28-51��)';        %%%%%%%
totalNum=classes*20;

data=zeros(rows,rows,totalNum);   %%data���ڴ洢ѹ�������ͼ������
labels=zeros(1,totalNum);   %labels�洢ͼ���Ӧ�����
% set=ones(1,totalNum);   %set����ѵ�����Ͳ��Լ� set=1��ʾѵ���� set=3��ʾ���Լ�

fileFolder=fullfile('E:\����',dir_name);
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';    %fileNames�洢��������30���������

numOfPic=0;  %���ڼ�¼ѹ�����ڼ���ͼ��

%���ζԸ�������ͼƬ���д���
for filenum= 3:32    
    files=dir(fullfile(fileFolder,dirOutput(filenum).name,'*_1.png'));   %%%%%%%
    fileNames2={files.name}';   %fileNames2�洢���ǵ�ǰ������к�׺Ϊ*_1.png���ļ�������
    num=size(files,1);
%     trainNum=num*3/4;
%     testNum=num/4;
    maindir=[fileFolder,'\',dirOutput(filenum).name];
    
    for i=61:num
        numOfPic=numOfPic+1;
        path=[maindir,'\',files(i).name];
        f=imread(path);
        im=imresize(f(:,:,1),[rows rows],'bilinear');  %��ά��256*256
        data(:,:,numOfPic)=im;        
    end    
    flag=flag+1 
end
labels(1:20) = 22;
labels(21:40) = 26;
labels(41:60) = 33;
labels(61:80) = 35;
labels(81:100) = 38;
labels(101:120) = 40;
labels(121:140) = 48;
labels(141:160) = 52;
labels(161:180) = 63;
labels(181:200) = 72;
labels(201:220) = 93;
labels(221:240) = 94;
labels(241:260) = 104;
labels(261:280) = 112;
labels(281:300) = 117;
labels(301:320) = 124;
labels(321:340) = 125;
labels(341:360) = 129;
labels(361:380) = 135;
labels(381:400) = 143;
labels(401:420) = 154;
labels(421:440) = 158;
labels(441:460) = 176;
labels(461:480) = 181;
labels(481:500) = 192;
labels(501:520) = 201;
labels(521:540) = 224;
labels(541:560) = 230;
labels(561:580) = 232;
labels(581:600) = 250;

%��data��labels��set�ϲ��洢Ϊimdb    
imdb.images.data = data;
imdb.images.labels = labels;
 
%�洢imdbΪmat��ʽ
dir='E:\����';
savefile=[dir,'\','Data','\',num2str(rows),'_40_large','.mat'];          %%%%%%
save(savefile,'imdb','-v7.3');