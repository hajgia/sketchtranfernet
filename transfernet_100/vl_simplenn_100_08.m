function res = vl_simplenn_100_08(net, x, dzdy, res, varargin)

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;

%Ç°Ïò
for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
    case 'normalize'
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'relu'
      res(i+1).x = vl_nnrelu(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'dropout'
      if opts.disableDropout
        res(i+1).x = res(i).x ;
      elseif opts.freezeDropout
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  if opts.conserveMemory & ~doder & i < numel(net.layers) - 1
    % TODO: forget unnecesary intermediate computations even when
    % derivatives are required
    res(i).x = [] ;
  end
%   if gpuMode & opts.sync
%     % This should make things slower, but on MATLAB 2014a it is necessary
%     % for any decent performance.
%     wait(gpuDevice) ;
%   end
  res(i).time = toc(res(i).time) ;
end

%·´Ïò£¬Çódzdx,dzdw£¬²¢¼ÇÂ¼·´ÏòÊ±¼äµÈ
if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:1
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'conv'
          if i ~= 1 || i~=4 || i~=7 || i~=9
              [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                  vl_nnconv(res(i).x, l.filters, l.biases, ...
                  res(i+1).dzdx, ...
                  'pad', l.pad, 'stride', l.stride) ;
          end
      case 'pool'
          if i == 13 
              res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
              'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
          end
%       case 'normalize'
%         res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
%       case 'softmax'
%         res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
%       case 'loss'
%         res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'relu'
          if i ~= 2 || i~=5 || i~=8 || i~=10
               res(i).dzdx = vl_nnrelu(res(i).x, res(i+1).dzdx) ;
          end
      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
      case 'dropout'    
               if opts.disableDropout
                    res(i).dzdx = res(i+1).dzdx ;
               else
                    res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
               end
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
%     if gpuMode & opts.sync
%       wait(gpuDevice) ;
%     end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end