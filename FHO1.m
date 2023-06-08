function [Best_Pos, Conv_History] = FHO1(X_train,X_test,y_train,y_test,VarNumber,VarMin,VarMax,MaxFes,nPop)
HN = randi([1 ceil(nPop/5)],1,1) ;      % Maximum number of FireHawks

% Counters
Iter=0;
FEs=0;

%% Initialization
Pop=[]; Cost=[];
for i=1:nPop
    % Initialize Positions
    Pop(i,:)=unifrnd(VarMin,VarMax,[1 VarNumber]);
    % Cost Function Evaluations
    M = py.mdn.mdgru(X_train,X_test,y_train,y_test,int32(Pop(i,1)),int32(Pop(i,2)),int32(Pop(i,3)),int32(Pop(i,4))).double;
    Cost(i,:)=M(1);
    FEs=FEs+1;
end

% Sort Population

[Cost, SortOrder]=sort(Cost);
Pop=Pop(SortOrder,:);
BestPop=Pop(1,:);
SP=mean(Pop);

% Fire Hawks
FHPops=Pop(1:HN,:);

% Prey
Pop2=Pop(HN+1:end,:);

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
GB=Cost(1);
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
        FHnear=FHPops(randi(HN),:);
        FHl_new=FHl+(Ir(1)*GB-Ir(2)*FHnear);
        FHl_new = max(FHl_new,VarMin);
        FHl_new = min(FHl_new,VarMax);
        PopTot=[PopTot ;FHl_new];
        
        for q=1:size(PR,1)
            
            Ir=unifrnd(0,1,1,2);
            PRq_new1=PR(q,:)+((Ir(1)*FHl-Ir(2)*SPl));
            PRq_new1 = max(PRq_new1,VarMin);
            PRq_new1 = min(PRq_new1,VarMax);
            PopTot=[PopTot ;PRq_new1];
            
            Ir=unifrnd(0,1,1,2);
            FHAlter=FHPops(randi(HN),:);
            PRq_new2=PR(q,:)+((Ir(1)*FHAlter-Ir(2)*SP));
            PRq_new2 = max(PRq_new2,VarMin);
            PRq_new2 = min(PRq_new2,VarMax);
            PopTot=[PopTot ;PRq_new2];
        end
    end
    for i=1:size(PopTot,1)
        M = py.mdn.mdgru(X_train,X_test,y_train,y_test,int32(PopTot(i,1)),int32(PopTot(i,2)),int32(PopTot(i,3)),int32(PopTot(i,4))).double;
        Cost(i,1)=M(1);
        FEs=FEs+1;
    end
    % Sort Population
    [Cost, SortOrder]=sort(Cost);
    PopTot=PopTot(SortOrder,:);
    Pop=PopTot(1:nPop,:);
    HN = randi([1 ceil(nPop/5)],1,1);
    BestPop=Pop(1,:);
    SP=mean(Pop);
    FHPops=Pop(1:HN,:);
    Pop2=Pop(HN+1:end,:);
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
    if Cost(1)<GB
        BestPos=BestPop;
    end
    GB=min(GB,Cost(1));
    % Store Best Cost Ever Found
    BestCosts(Iter,1)=GB;
    % Show Iteration Information
end
Eval_Number=FEs;
Conv_History=BestCosts;
Best_Pos=BestPop;
end






