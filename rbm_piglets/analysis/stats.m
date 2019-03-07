clear all;

%% Load activation maps
nii=load_untouch_nii('norm/normalize_Smoothed125_ThresZ1_rbm_piglets_weights_reconstructed_Corrected_last.nii');
header=nii.hdr;
img=double(nii.img);
clear nii;
img(isnan(img))=0;

%% Load RSN atlas
a=load_untouch_nii('piglet_analysis/pig_atlas/Pig_RSN_Atlas.nii');
at=double(a.img);

%% Load Anatomy to create mask
ma=load_untouch_nii('piglet_analysis/normalize_Masked_Volume_Corrected.nii');
anat=double(ma.img);
mask=anat;
mask(mask>0)=1;
at=at.*mask;
atab=at(at>0);
at_tab=tabulate(atab);

%% Calculate stats
at_totals=tabulate(reshape(at,[size(at,1)*size(at,2)*size(at,3) 1]));
for i=1:size(img,4)
    img_temp=img(:,:,:,i);
    avgs_total(i)=mean(img_temp(mask>0));
    imgc=reshape(img_temp,[size(at,1)*size(at,2)*size(at,3) 1]);
    region=at(img_temp>0);
    for j=1:size(at_totals,1)
        count=region(region==at_totals(j,1));
        region_totals(j,i)=size(count,1);
        count_outside=region(region~=at_totals(j,1));
        outside_region_totals(j,i)=size(count_outside,1);
        
        at_temp=at;
        at_temp(at~=at_totals(j,1))=0;
        avgs_region(j)=mean(img_temp(at_temp>0));
        at_temp=reshape(at_temp,[size(at,1)*size(at,2)*size(at,3) 1]);
        pearson(j,i)=corr(imgc,at_temp);
    end
    region_per(:,i)=region_totals(:,i)./at_totals(:,2);
    ratio(:,i)=avgs_region/avgs_total(i);
    R(:,i)=region_totals(:,i)./(outside_region_totals(:,i)+at_totals(:,2));
end

%% Find maps with maximal Pearson values
for i=1:size(at_totals,1)
    [maxr_temp, indr_temp] = max(pearson(i,:));
    maxs(2,i)=maxr_temp;
    maxs(1,i)=indr_temp;
    maxs(3,i)=ratio(i,indr_temp);
    
    maxs(5,i)=region_totals(i,indr_temp);
    maxs(6,i)=region_per(i,indr_temp);
end


save('statsMaxs.txt','maxs','-ascii','-tabs')
