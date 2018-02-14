load('phoneticsLabels5.mat');
%fileIDP = fopen('mapP.txt','w');
fileIDV = fopen('mapVN.txt','w');
timePointer=2;

%%

for fileI = 1:size(fileNamesNew,2)
    if(exist(fileNamesNew(fileI)))
       videoLength=VideoSize(fileI);
       load(fileNamesNew(fileI));
       mouthLength=size(ROIs{1,1},2);
       offset=videoLength-mouthLength;
       fileFolder=strsplit(fileNamesNew(fileI),'/')
       folderName=fileFolder(1)+'/'+fileFolder(2);

       fileName=strtok(fileFolder(3),'.');
       starT(timePointer)=starT(timePointer)+offset;
       while(starT(timePointer)~=-1)
           startIndex=starT(timePointer)-offset;
           endIndex=endT(timePointer)-offset;
           if(endIndex<startIndex+3||startIndex<1||endIndex>mouthLength)
                timePointer=timePointer+1;
                continue;
           end
           for j=startIndex+3:endIndex

               midIndex=round((startIndex+endIndex)/2);

               %%endImage=imresize(ROIs{1,1}{1,endIndex},[224,224]);
%                ImageJ=imresize(ROIs{1,1}{1,j},[224,224]);
%                ImageJM1=imresize(ROIs{1,1}{1,j-1},[224,224]);
%                ImageJM2=imresize(ROIs{1,1}{1,j-2},[224,224]);
%                ImageJM3=imresize(ROIs{1,1}{1,j-3},[224,224]);
%                ImageO=cat(3, ImageJ, ImageJM1, ImageJM2);
%                ImageO2=cat(3, ImageJ-ImageJM1, ImageJM1-ImageJM2, ImageJM2-ImageJM3);
% 
%                 ImageO2=ImageO2+0.5;
              % imagesc(diffImage);

               %ImageName=strcat(folderName,'O/',fileName,sprintf('-%03d',j),'.png');
               ImageNameN=strcat(folderName,'O/',fileName,sprintf('-NNN%03d',j),'.png');
%                imwrite(ImageO2,convertStringsToChars(ImageName));
%                imwrite(ImageO,convertStringsToChars(ImageNameN));
               %fprintf(fileIDP,'%s %s\n',ImageName,num2str(labelN(timePointer)));
               fprintf(fileIDV,'%s %s\n',ImageNameN,num2str(labelTV(timePointer)));
           end
            timePointer=timePointer+1;
            
       end
       timePointer=timePointer+2;
    end

    
end
fclose(fileIDP);
fclose(fileIDV);

% %%
% for i=1:143
%     imshow(ROIs{1,1}{1,i});
%     pause(3);
% end
%%
% folderNames=["Mercury"];
% for fileI = 1:size(fileNamesNew,2)
% 
%        fileFolder=strsplit(fileNamesNew(fileI),'/');
%        folderNames(end+1)=fileFolder(2);
% end
% folderNames=unique(folderNames)';
% 
% %%
% labelTV2=labelN;
% for i=1:216084
%     if(labelT(i)=='')
%         labelTV2(i)=-1;
%     else
%         labelTV2(i)=find(WWlist==p2v(find((labelT(i)==p2v(:,1))),2));
%     end
% 
% end
% i

%%
load('mapStartEndFiles');
for k=1:5
    k
    mapSERand=mapStartEndFiles(randperm(58),:);
    mapPNTest=mapPN(mapSERand(1,1):mapSERand(1,2),:);
    mapVNTest=mapVN(mapSERand(1,1):mapSERand(1,2),:);
    for i=2:9
        mapPNTest=cat(1,mapPNTest,mapPN(mapSERand(i,1):mapSERand(i,2),:));
        mapVNTest=cat(1,mapVNTest,mapVN(mapSERand(i,1):mapSERand(i,2),:));
    end

    mapPNTrain=mapPN(mapSERand(10,1):mapSERand(10,2),:);
    mapVNTrain=mapVN(mapSERand(10,1):mapSERand(10,2),:);
    for i=11:58
        mapPNTrain=cat(1,mapPNTrain,mapPN(mapSERand(i,1):mapSERand(i,2),:));
        mapVNTrain=cat(1,mapVNTrain,mapVN(mapSERand(i,1):mapSERand(i,2),:));
    end

    fileID = fopen(strcat('mapPNTrainF',num2str(k),'.txt'),'w');
    fprintf(fileID,'%s\t%s\n',mapPNTrain');
    fclose(fileID);
    
    fileID = fopen(strcat('mapVNTrainF',num2str(k),'.txt'),'w');
    fprintf(fileID,'%s\t%s\n',mapVNTrain');
    fclose(fileID);
    
    fileID = fopen(strcat('mapPNTestF',num2str(k),'.txt'),'w');
    fprintf(fileID,'%s\t%s\n',mapPNTest');
    fclose(fileID);
    
    fileID = fopen(strcat('mapVNTestF',num2str(k),'.txt'),'w');
    fprintf(fileID,'%s\t%s\n',mapVNTest');
    fclose(fileID);
end