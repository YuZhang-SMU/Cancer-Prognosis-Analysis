clear
path1='..\'; % File location of the cropped pathes 
files1=dir(path1);
Imatch=imread('Reference image.tiff');
for n=3:numel(files1)
    path2=[path1,files1(n).name];
    path3=['..\',files1(n).name,'\']; % File location of pathes after histogram matching 
    mkdir(path3);
    files2=dir(path2);
    for i=3:numel(files2)
        I=imread([path2,'\',files2(i).name]);
        R=I(:,:,1);
        G=I(:,:,2);
        B=I(:,:,3);
        Rmatch=Imatch(:,:,1);
        Gmatch=Imatch(:,:,2);
        Bmatch=Imatch(:,:,3);
        Rmatch_hist=imhist(Rmatch);
        Gmatch_hist=imhist(Gmatch);
        Bmatch_hist=imhist(Bmatch);
        Rout=histeq(R,Rmatch_hist);
        Gout=histeq(G,Gmatch_hist);
        Bout=histeq(B,Bmatch_hist);
        img(:,:,1)=Rout;
        img(:,:,2)=Gout;
        img(:,:,3)=Bout;
        imwrite(img,[path3,files2(i).name]);
    end
end