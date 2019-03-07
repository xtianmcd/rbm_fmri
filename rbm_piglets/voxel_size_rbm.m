clear all;

% Load single pig fmri (only for header purposes)
fmri=load_untouch_nii('masked_normalize_sliced_unwarp_fmri.nii');
header=fmri.hdr;

% Load mask
m=load_untouch_nii('12_pigsMask.nii');
mask=m.img;

% Load activation maps
% act=load_untouch_nii('Group_Temporal_GIFTMask_DL_292_Lam075.nii');
act=load_untouch_nii('rpw/rbm_piglets_weights_reconstructed_last.nii');

img=act.img;

img(img<0)=0;
% Preprocess data
for a=1:size(img,4)
%for a=1:1
    aimg=squeeze(img(:,:,:,a));   
    % Find max of histgram
    himg=aimg(aimg>0); 
    if max(himg~=0)
    bins=linspace(min(himg(:)),max(himg(:)),101);
    h = histcounts(himg(:),bins);
    [max_count, max_loc]=max(h);
    
    % Create Gaussian distribution
    hgimg=aimg(aimg>bins(max_loc+1));
    gimg=[hgimg;-hgimg+2*bins(max_loc+1)];
    
    % Remove outliers
    outlier=4*std(gimg)+mean(gimg);
    aimg(aimg>outlier)=0;
    
    % Threshold
    aimg=aimg/max(aimg(:));
    %aimg(aimg<0.35)=0;
    himg=aimg(mask>0);
    %figure;
    %histogram(himg);
    %in=himg(himg>0.35);
    %per=size(in)/size(himg)
    %sorted=sort(himg);
    %thres=sorted(int16(size(himg)*.85));
    thres=std(himg)+mean(himg);
    aimg(aimg<thres(1))=0;
    %z=(thres(1)-mean(himg))/std(himg);
    
    
    % Normalize and Guassian Smooth 
    aimg=imgaussfilt3(aimg,1.25);
    amax=max(aimg(:));
    aimg=aimg/amax;
    img(:,:,:,a)=aimg;
    end
    
    % Save individual files
%     anat.img=aimg;
%     fil=sprintf('rbm_piglets_weights_reconstructed_corrected%d.nii',a);
%     save_untouch_nii(anat,fil);

    % Save 2D images
%     f=figure('visible', 'off');
%     for s=4:size(aimg,3)-3
%         subplot(7,4,s);
%         imshow(aimg(:,:,s),[]);
%     end
%     fil=sprintf('Atom%d.tif',a);
%     saveas(f,fil);
%     close(f);
end
act.img=img;

% Fix header (make same as mask header, except 4D)
act.hdr=header;
act.hdr.dime.dim(1)=4;
act.hdr.dime.dim(5)=size(img,4);
act.hdr.dime.datatype=16;
act.hdr.dime.bitpix=32;

%save_untouch_nii(act,'Smoothed125_ThresZ1_rbm_piglets_weights_reconstructed_Corrected_{}_{}.nii');
save_untouch_nii(act,'Smoothed125_ThresZ1_rbm_piglets_weights_reconstructed_Corrected_last.nii');


