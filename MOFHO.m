function [Best_Pos, Conv_History] = MOFHO(X_train,X_test,y_train,y_test,dim,lb,ub,MaxFes,nPop)
Iter=0;
FEs=0;
Pop=[]; Cost=[];
for i=1:nPop
    % Initialize Positions
    Pop(i,:)=unifrnd(lb,ub,[1 dim]);
    % Cost Function Evaluations
    Cost(i,:)=py.mdn.mdgru(X_train,X_test,y_train,y_test,int32(Pop(i,1)),int32(Pop(i,2)),int32(Pop(i,3)),int32(Pop(i,4))).double;
    FEs=FEs+1;
end
% Sort Population
obj_no = 2;
ArchiveMaxSize=200;
Archive_X=zeros(200,dim);
Archive_F=ones(200,obj_no)*inf;
Archiveember_no=0;
[Archive_X, Archive_F, Archiveember_no]=UpdateArchive(Archive_X, Archive_F, Pop, Cost, Archiveember_no);
if Archiveember_no>ArchiveMaxSize
    Archiveem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    [Archive_X, Archive_F, Archiveem_ranks, Archiveember_no]=HandleFullArchive(Archive_X, Archive_F, Archiveember_no, Archiveem_ranks, ArchiveMaxSize);
else
    Archiveem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
end
Archiveem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
index=RouletteWheelSelection(1./Archiveem_ranks);
if index==-1
    index=1;
end
fitnessBest=Archive_F(index,:);
BestPop=Archive_X(index,:)';
SP=mean(Pop);

% [Cost, SortOrder]=sort(Cost);
% Pop=Pop(SortOrder,:);
% BestPop=Pop(1,:);
% SP=mean(Pop);
HN = randi([1 ceil(nPop/5)],1,1) ;      % Maximum number of FireHawks
% Fire Hawks
if HN<size(Archive_X,1)
    FHPops=Archive_X(1:HN,:);
else
    FHPops=Archive_X;
    HN = size(FHPops,1);
end
% Prey
Pop2=Pop(find(ismember(Pop(:,1),FHPops(:,1))==0),:);

% Distance between  Fire Hawks and Prey
for i=1:HN
    nPop2=size(Pop2,1);
    if nPop2<HN
        break
    end
    Dist=[];
    for q=1:nPop2
        Dist(q,1)=distance(FHPops(i,:), Pop2(q,:));
    end
    [ ~, b]=sort(Dist);
    alfa=randi(nPop2);
    PopNew{i,:}=Pop2(b(1:alfa),:);
    Pop2(b(1:alfa),:)=[];
    if isempty(Pop2)==1
        break
    end
end
if isempty(Pop2)==0
    PopNew{end,:}=[PopNew{end,:} ;Pop2];
end

% Update Bests
GB=fitnessBest;
BestPos=BestPop;

%% Main Loop
while FEs<MaxFes
    Iter=Iter+1;
    PopTot=[];
    Cost=[];
    for i=1:size(PopNew,1)
        PR=cell2mat(PopNew(i));
        FHl=FHPops(i,:);
        SPl=mean(PR);
        
        Ir=unifrnd(0,1,1,2);
        FHnear=FHPops(randi(size(FHPops,1)),:);
        FHl_new=FHl+(Ir(1)*sum(GB)-Ir(2)*FHnear);
        FHl_new = max(FHl_new,lb);
        FHl_new = min(FHl_new,ub);
        PopTot=[PopTot ;FHl_new];
        
        for q=1:size(PR,1)
            
            Ir=unifrnd(0,1,1,2);
            PRq_new1=PR(q,:)+((Ir(1)*FHl-Ir(2)*SPl));
            PRq_new1 = max(PRq_new1,lb);
            PRq_new1 = min(PRq_new1,ub);
            PopTot=[PopTot ;PRq_new1];
            
            Ir=unifrnd(0,1,1,2);
            FHAlter=FHPops(randi(size(FHPops,1)),:);
            PRq_new2=PR(q,:)+((Ir(1)*FHAlter-Ir(2)*SP));
            PRq_new2 = max(PRq_new2,lb);
            PRq_new2 = min(PRq_new2,ub);
            PopTot=[PopTot ;PRq_new2];
        end
    end
    for i=1:size(PopTot,1)
        Cost(i,:)=py.mdn.mdgru(X_train,X_test,y_train,y_test,int32(PopTot(i,1)),int32(PopTot(i,2)),int32(PopTot(i,3)),int32(PopTot(i,4))).double;
        FEs=FEs+1;
    end
    % Sort Population
    [Archive_X, Archive_F, Archiveember_no]=UpdateArchive(Archive_X, Archive_F, PopTot, Cost, Archiveember_no);
    if Archiveember_no>ArchiveMaxSize
        Archiveem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        [Archive_X, Archive_F, Archiveem_ranks, Archiveember_no]=HandleFullArchive(Archive_X, Archive_F, Archiveember_no, Archiveem_ranks, ArchiveMaxSize);
    else
        Archiveem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end
    Archiveem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    index=RouletteWheelSelection(1./Archiveem_ranks);
    if index==-1
        index=1;
    end
    fitnessBest=Archive_F(index,:);
    BestPop=Archive_X(index,:)';

    HN = randi([1 ceil(nPop/5)],1,1) ;      % Maximum number of FireHawks
    % Fire Hawks
    if HN<size(Archive_X,1)
        FHPops=Archive_X(1:HN,:);
    else
        FHPops=Archive_X;
        HN = size(FHPops,1);
    end

    [Cost, SortOrder]=sort(Cost);
    PopTot=PopTot(SortOrder,:);
    Pop=PopTot(1:nPop,:);
    SP=mean(Pop);
    Pop2=Pop(find(ismember(Pop(:,1),FHPops(:,1))==0),:);
    clear PopNew
    
    % Distance between  Fire Hawks and Prey
    for i=1:HN
        nPop2=size(Pop2,1);
        if nPop2<HN
            break
        end
        Dist=[];
        for q=1:nPop2
            Dist(q,1)=distance(FHPops(i,:), Pop2(q,:));
        end
        [ ~, b]=sort(Dist);
        alfa=randi(nPop2);
        PopNew{i,:}=Pop2(b(1:alfa),:);
        Pop2(b(1:alfa),:)=[];
        if isempty(Pop2)==1
            break
        end
    end
    if isempty(Pop2)==0
        PopNew{end,:}=[PopNew{end,:} ;Pop2];
    end
    % Update Bests
    if dominates(fitnessBest,GB)
        BestPos=BestPop;
        GB = fitnessBest;
    end
    % Store Best Cost Ever Found
    BestCosts(Iter,:)=GB;
    % Show Iteration Information
    disp(['Iteration ' num2str(Iter) ': Best Cost = ' num2str(BestCosts(Iter,:))]);
end
Eval_Number=FEs;
Conv_History=BestCosts;
Best_Pos=BestPop;
end

