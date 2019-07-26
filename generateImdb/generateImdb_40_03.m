%���command window and workspace
clear all;
close all;
clc;

flag=0;  %���ڹ۲�ѵ�����ڼ����ļ���
rows=256;  %ѹ����ͼ���С
classes=50;   %��ֵ��ʾ��ǰҪ��������ĸ�������Ϊ�������������ᵼ��out of memory
dir_name='E:\����\svg-5strokes';
totalNum=classes*80;

data=zeros(rows,rows,totalNum);   %%data���ڴ洢ѹ�������ͼ������
labels=zeros(1,totalNum);   %labels�洢ͼ���Ӧ�����
set=ones(1,totalNum);   %set����ѵ�����Ͳ��Լ� set=1��ʾѵ���� set=3��ʾ���Լ�

fileFolder=fullfile(dir_name);
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';    %fileNames�洢��������250���������

numOfPic=0;  %���ڼ�¼ѹ�����ڼ���ͼ��

%���ζԸ�������ͼƬ���д���
for filenum= 103:152    
    files=dir(fullfile(dir_name,dirOutput(filenum).name,'*_1.png'));   
    fileNames2={files.name}';   %fileNames2�洢���ǵ�ǰ������к�׺Ϊ*_1.png���ļ�������
    num=size(files,1);
    trainNum=num*3/4;
    testNum=num/4;
    maindir=[dir_name,'\',dirOutput(filenum).name];
    
    for i=1:num
        numOfPic=numOfPic+1;
        path=[maindir,'\',files(i).name];
        f=imread(path);
        im=imresize(f(:,:,1),[rows rows],'bilinear');  %��ά��256*256
        data(:,:,numOfPic)=im;
        
        labels(1,(filenum-103)*80+i)=filenum-2;
           
        if i>trainNum
            set(1,(filenum-103)*80+i)=3;
        end
    end    
    flag=flag+1 
end

%��data��labels��set�ϲ��洢Ϊimdb    
imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = set;

imdb.meta.sets = {'train', 'val', 'test' } ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:classes,'uniformoutput',false) ;

%�洢imdbΪmat��ʽ
dir='E:\����';
savefile=[dir,'\','Data','\',num2str(rows),'_40_03','.mat'];
save(savefile,'imdb','-v7.3');
