function []=cnn_sketch(varargin)
%one pic on deconvnet
run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir=fullfile('data','sketch');
opts.expDir=fullfile('data','sketch-baseline');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------


  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;


% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
rows=128;
load(fullfile(opts.dataDir,'sketch.mat'));

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),rows,rows,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
