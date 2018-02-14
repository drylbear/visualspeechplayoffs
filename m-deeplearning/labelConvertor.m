length=size(labelT,1);
labelN=zeros(length,1);
for i=1:length
    if(labelT(i)~='')
        labelN(i)=find(ismember(beepList,labelT(i)));
    else
        labelN(i)=-1;
    end
    
end

%%
fileNamesNew2=fileNamesNew;
for i=1:size(fileNamesNew,1)
    fileNamesNew2(i)=strjoin(strsplit(fileNamesNew(i),'Clips/straightcam/'),'');

end
%%
for i=1:size(labelTV,1)
    [ia, ib] = find(ismember(A(:,1),labelT(i)));
    labelTV(i)=find(ismember(WWlist,A(ia,2)));
    

end