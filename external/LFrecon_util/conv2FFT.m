function y = conv2FFT(x, m)
global zeroImageEx;
global exsize;

xsize = size(x);
msize = size(m);
fsize = xsize;
mmid = floor(msize/2);

x = zeroPad(x, zeroImageEx);
m = zeroPad(m, zeroImageEx);
mc = 1 + mmid;
me = mc + exsize - 1;
m = exindex(m, mc(1):me(1), mc(2):me(2), 'circular');

y = real(ifft2(fft2(x) .* fft2(m)));   
y = y(1:fsize(1), 1:fsize(2));

end
