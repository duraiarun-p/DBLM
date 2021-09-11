function [Z1a,IP1a,IT1a,MP2a,MT2a]=blarloadpycall(directory_name1,directory_name2)
[IP1a,IT1a,MP1a,MT1a,Z1a,~,SN1a,SN2a]=loadtwomedimagadatawithmask4woresZ_inbuilt(directory_name1,directory_name2,'CT');
Sname='Structure 2';
% TF1x=ismember(SN1a,Sname);
% TF2x=ismember(SN2a,Sname);
% ID1x=find(TF1x);
% ID2x=find(TF2x);
% MP2=MP1{ID1x,1};
% MT2=MT1{ID2x,1};
ID1x=find(contains(SN1a,Sname));
ID2x=find(contains(SN2a,Sname));
MP2a=MP1a{ID1x,1};
MT2a=MT1a{ID2x,1};
% [IP1a,IT1a,MP2a,MT2a,Z1a,ZLa]=rearrangeslice(IP1a,IT1a,MP2a,MT2a,Z1a); % Common no. of slices selection
%     if flag==0
%     [sr,sc,sp]=size(IP1a);
%     IP1a=randi(2000,[sr,sc,sp],'int16');
%     MP2a=false(sr,sc,sp);
%     end

% [Cr,Rel2,Rel1,DSC2,DSC1]=BLMIR3DDiceFx_24_inbuilt(Z1a,IP1a,IT1a,MP2a,MT2a,indSize,mbSize,p,lambda1A,lambda2,m,n,LenX,LenY,costV);%1 lambda
end
%%
function [im1,im2,MaskV1,MaskV2,Z1,ZL,SN1,SN2]=loadtwomedimagadatawithmask4woresZ_inbuilt(directory_name1,directory_name2,Imgmodal)
%Load data 1 with structure names
myhandle1=Open_DicomGui();
[~,~]=DG_load_data(myhandle1,directory_name1,Imgmodal);
[SN1,~] = DG_get_names(myhandle1);
SN1=SN1';
[im1,Z1,~]=DG_get_image_data(myhandle1);

%Load data 2 with structure names
myhandle2=Open_DicomGui();
[~,~]=DG_load_data(myhandle2,directory_name2,Imgmodal);
[SN2,~] = DG_get_names(myhandle2);
SN2=SN2';
[im2,~,~]=DG_get_image_data(myhandle2);

[~,~,ZL]=size(im1);

SN1L=length(SN1);
SN2L=length(SN2);

MaskV1=cell(SN1L,1);
for contouri=1:SN1L
    [MaskV1{contouri,1},~]=DG_generate_volume_mask(myhandle1, contouri);
end

MaskV2=cell(SN2L,1);
for contouri=1:length(SN2)
    [MaskV2{contouri,1},~]=DG_generate_volume_mask(myhandle2, contouri);
end
Close_DicomGui(myhandle1);
Close_DicomGui(myhandle2);
end