function IMG_Rect = ImageRect(IMG_BW, xCenter, yCenter, dx, M, Crop, XcutLeft, XcutRight, YcutUp, YcutDown)
%% Resample
Xresample = [fliplr((xCenter+1):-dx/M:1)  ((xCenter+1)+dx/M:dx/M:size(IMG_BW,2))];
Yresample = [fliplr((yCenter+1):-dx/M:1)  ((yCenter+1)+dx/M:dx/M:size(IMG_BW,1))];
[X, Y] = meshgrid( (1:1:size(IMG_BW,2)),   (1:1:size(IMG_BW,1)) ); 
[Xq,Yq] = meshgrid( Xresample , Yresample );

Mdiff = floor(M/2);
XqCenterInit = find(Xq(1,:)==(xCenter+1)) - Mdiff;
XqInit = XqCenterInit -  M*floor(XqCenterInit/M)+M;
YqCenterInit = find(Yq(:,1)==(yCenter+1)) - Mdiff;
YqInit = YqCenterInit -  M*floor(YqCenterInit/M)+M;

XresampleQ = Xresample(XqInit:end);
YresampleQ = Yresample(YqInit:end);
[Xqq,Yqq] = meshgrid( XresampleQ , YresampleQ );

IMG_RESAMPLE = interp2( X, Y,  IMG_BW, Xqq, Yqq );
IMG_RESAMPLE_crop1 = IMG_RESAMPLE( (1:1:M*floor((size(IMG_RESAMPLE,1)-YqInit)/M)), (1:1:M*floor((size(IMG_RESAMPLE,2)-XqInit)/M)) );

%% Crop
if Crop
    XsizeML = size(IMG_RESAMPLE_crop1,2)/M;
    YsizeML = size(IMG_RESAMPLE_crop1,1)/M;
    if (XcutLeft + XcutRight)>=XsizeML
        error('X-cut range is larger than the x-size of image');
    end
    if (YcutUp + YcutDown)>=YsizeML
        error('Y-cut range is larger than the y-size of image');
    end

    Xrange = (1+XcutLeft:XsizeML-XcutRight);
    Yrange = (1+YcutUp:YsizeML-YcutDown);
    
    IMG_RESAMPLE_crop2 = IMG_RESAMPLE_crop1(   ((Yrange(1)-1)*M+1 :Yrange(end)*M),  ((Xrange(1)-1)*M+1 :Xrange(end)*M) );
else
    IMG_RESAMPLE_crop2 = IMG_RESAMPLE_crop1;
end

IMG_Rect = IMG_RESAMPLE_crop2;