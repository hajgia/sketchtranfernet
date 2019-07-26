%���command window and workspace
clear all;
close all;
clc;

flag=0;  %���ڹ۲�ѵ�����ڼ����ļ���
rows=256;  %ѹ����ͼ���С
classes=30;   %��ֵ��ʾ��ǰҪ��������ĸ�������Ϊ�������������ᵼ��out of memory
dir_name='svg-5strokes-�бʻ�(15-18��)';        %%%%%%%
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
    files=dir(fullfile(fileFolder,dirOutput(filenum).name,'*_4.png'));   %%%%%%%
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

%��data��labels��set�ϲ��洢Ϊimdb    
imdb.images.data = data;
imdb.images.labels = labels;
 
%�洢imdbΪmat��ʽ
dir='E:\����';
savefile=[dir,'\','Data','\',num2str(rows),'_100_middle','.mat'];          %%%%%%
save(savefile,'imdb','-v7.3');