%清除command window and workspace
clear all;
close all;
clc;

flag=0;  %用于观察训练到第几个文件夹
rows=256;  %压缩后图像大小
classes=50;   %该值表示当前要处理的类别的个数，因为数据类别数过大会导致out of memory
dir_name='E:\刘杨\svg-5strokes';
totalNum=classes*80;

data=zeros(rows,rows,totalNum);   %%data用于存储压缩过后的图像数据
labels=zeros(1,totalNum);   %labels存储图像对应的类别
set=ones(1,totalNum);   %set区别训练集和测试集 set=1表示训练集 set=3表示测试集

fileFolder=fullfile(dir_name);
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';    %fileNames存储的是所有250个类的类名

numOfPic=0;  %用于记录压缩到第几张图像

%依次对各个类别的图片进行处理
for filenum= 103:152    
    files=dir(fullfile(dir_name,dirOutput(filenum).name,'*_1.png'));   
    fileNames2={files.name}';   %fileNames2存储的是当前类的所有后缀为*_1.png的文件的名字
    num=size(files,1);
    trainNum=num*3/4;
    testNum=num/4;
    maindir=[dir_name,'\',dirOutput(filenum).name];
    
    for i=1:num
        numOfPic=numOfPic+1;
        path=[maindir,'\',files(i).name];
        f=imread(path);
        im=imresize(f(:,:,1),[rows rows],'bilinear');  %降维到256*256
        data(:,:,numOfPic)=im;
        
        labels(1,(filenum-103)*80+i)=filenum-2;
           
        if i>trainNum
            set(1,(filenum-103)*80+i)=3;
        end
    end    
    flag=flag+1 
end

%将data，labels和set合并存储为imdb    
imdb.images.data = data;
imdb.images.labels = labels;
imdb.images.set = set;

imdb.meta.sets = {'train', 'val', 'test' } ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:classes,'uniformoutput',false) ;

%存储imdb为mat格式
dir='E:\刘杨';
savefile=[dir,'\','Data','\',num2str(rows),'_40_03','.mat'];
save(savefile,'imdb','-v7.3');
