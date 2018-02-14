load('phoneticsLabels3.mat');
fileIDP = fopen('mapP.txt','w');
fileIDV = fopen('mapV.txt','w');
timePointer=2;

%%

for fileI = 1:size(fileNamesNew,1)
    if(exist(fileNamesNew(fileI)))
       videoLength=VideoSize(fileI);
       load(fileNamesNew(fileI));
       mouthLength=size(ROIs{1,1},2);
       offset=videoLength-mouthLength;
       fileFolder=strsplit(fileNamesNew(fileI),'/')
       folderName=fileFolder(1);

       fileName=strtok(fileFolder(2),'.');
       startT(timePointer)=startT(timePointer)+offset;
       while(startT(timePointer)~=-1)
           startIndex=startT(timePointer)-offset;
           endIndex=endT(timePointer)-offset;
           if(endIndex<startIndex+2||startIndex<1||endIndex>mouthLength)
                timePointer=timePointer+1;
                continue;
           end
           midIndex=round((startIndex+endIndex)/2);
           
           %%endImage=imresize(ROIs{1,1}{1,endIndex},[224,224]);
           startImage=imresize(ROIs{1,1}{1,startIndex},[224,224]);
           midImage=imresize(ROIs{1,1}{1,midIndex},[224,224]);
           
           diffImage=-startImage+midImage;
           diffImage=diffImage-min(diffImage(:));
           diffImage=diffImage./max(diffImage(:));
          % imagesc(diffImage);
           
           ImageName=strcat(folderName,'O/',fileName,sprintf('-%03d',midIndex),'.png');
           ImageNameN=strcat(folderName,'O/',fileName,sprintf('-NNN%03d',midIndex),'.png');
           imwrite(diffImage,convertStringsToChars(ImageName));
           imwrite(midImage,convertStringsToChars(ImageNameN));
           fprintf(fileIDP,'%s %s\n',ImageName,num2str(labelN(timePointer)));
           fprintf(fileIDV,'%s %s\n',ImageName,num2str(labelTV(timePointer)));
            timePointer=timePointer+1;
       end
       timePointer=timePointer+2;
    end

    
end
fclose(fileIDP);
fclose(fileIDV);

%%
for i=1:143
    imshow(ROIs{1,1}{1,i});
    pause(3);
end
