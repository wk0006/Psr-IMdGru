function [Xfood, fval] = MOSO(X_train,X_test,y_train,y_test,N,T,dim,lb,ub)
%initial 
vec_flag=[1,-1];
Threshold=0.25;
Thresold2= 0.6;
C1=0.5;
C2=.05;
C3=2;
X=lb+rand(N,dim).*(ub-lb);
GYbest = inf;
obj_no = 2;
Archive_X=zeros(200,dim);
Archive_F=ones(200,obj_no)*inf;
Archive_member_no=0;
for i=1:N
    fitness(i,:)=py.mdn.mdlstm(X_train,X_test,y_train,y_test,X(i,1),X(i,2),X(i,3)).double;
    if dominates( fitness(i,:),GYbest)
        GYbest=fitness(i,:);
        Xfood=X(i,:);
    end
end
% [GYbest, gbest] = min(fitness);
% Xfood = X(gbest,:);
ArchiveMaxSize=200;
Archive_X_m=zeros(200,dim);
Archive_F_m=ones(200,obj_no)*inf;
Archive_member_no_m=0;
%Diving the swarm into two equal groups males and females
Nm=round(N/2);%eq.(2&3)
Xm=X(1:Nm,:);
fitness_m=fitness(1:Nm,:);
[Archive_X_m, Archive_F_m, Archive_member_no_m]=UpdateArchive(Archive_X_m, Archive_F_m, Xm, fitness_m, Archive_member_no_m);
if Archive_member_no_m>ArchiveMaxSize
    Archive_mem_ranks_m=RankingProcess(Archive_F_m, ArchiveMaxSize, obj_no);
    [Archive_X_m, Archive_F_m, Archive_mem_ranks_m, Archive_member_no_m]=HandleFullArchive(Archive_X_m, Archive_F_m, Archive_member_no_m, Archive_mem_ranks_m, ArchiveMaxSize);
else
    Archive_mem_ranks_m=RankingProcess(Archive_F_m, ArchiveMaxSize, obj_no);
end
Archive_mem_ranks_m=RankingProcess(Archive_F_m, ArchiveMaxSize, obj_no);
index_m=RouletteWheelSelection(1./Archive_mem_ranks_m);
if index_m==-1
    index_m=1;
end
fitnessBest_m=Archive_F_m(index_m,:);
Xbest_m=Archive_X_m(index_m,:)';

Archive_X_f=zeros(200,dim);
Archive_F_f=ones(200,obj_no)*inf;
Archive_member_no_f=0;
Nf=N-Nm;
Xf=X(Nm+1:N,:);
fitness_f=fitness(Nm+1:N,:);
[Archive_X_f, Archive_F_f, Archive_member_no_f]=UpdateArchive(Archive_X_f, Archive_F_f, Xf, fitness_f, Archive_member_no_f);
if Archive_member_no_f>ArchiveMaxSize
    Archive_mem_ranks_f=RankingProcess(Archive_F_f, ArchiveMaxSize, obj_no);
    [Archive_X_f, Archive_F_f, Archive_mem_ranks_f, Archive_member_no_f]=HandleFullArchive(Archive_X_f, Archive_F_f, Archive_member_no_f, Archive_mem_ranks_f, ArchiveMaxSize);
else
    Archive_mem_ranks_f=RankingProcess(Archive_F_f, ArchiveMaxSize, obj_no);
end
Archive_mem_ranks_f=RankingProcess(Archive_F_f, ArchiveMaxSize, obj_no);
index_f=RouletteWheelSelection(1./Archive_mem_ranks_f);
if index_f==-1
    index_f=1;
end
fitnessBest_f=Archive_F_f(index_f,:);
Xbest_f=Archive_X_f(index_f,:)';
% [fitnessBest_m, gbest1] = min(fitness_m);
% Xbest_m = Xm(gbest1,:);
% [fitnessBest_f, gbest2] = min(fitness_f);
% Xbest_f = Xf(gbest2,:);

for t = 1:T
    Temp=exp(-((t)/T));  %eq.(4)
    Q=C1*exp(((t-T)/(T)));%eq.(5)
    if Q>1        Q=1;    end
    % Exploration Phase (no Food)
    if Q<Threshold
        for i=1:Nm
            for j=1:1:dim
                rand_leader_index = floor(Nm*rand()+1);
                X_randm = Xm(rand_leader_index, :);
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                Am=exp(-(0.5*fitness_m(rand_leader_index,1)+0.5*fitness_m(rand_leader_index,2))/((0.5*fitness_m(i,1)+0.5*fitness_m(i,2))+eps));%eq.(7)
                Xnewm(i,j)=X_randm(j)+Flag*C2*Am*((ub(j)-lb(j))*rand+lb(j));%eq.(6)
            end
        end
        for i=1:Nf
            for j=1:1:dim
                rand_leader_index = floor(Nf*rand()+1);
                X_randf = Xf(rand_leader_index, :);
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                Af=exp(-(0.5*fitness_f(rand_leader_index,1)+0.5*fitness_f(rand_leader_index,2))/((0.5*fitness_f(i,1)+0.5*fitness_f(i,2))+eps));%eq.(9)
                Xnewf(i,j)=X_randf(j)+Flag*C2*Af*((ub(j)-lb(j))*rand+lb(j));%eq.(8)
            end
        end
    else %Exploitation Phase (Food Exists)
        if Temp>Thresold2  %hot
            for i=1:Nm
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                for j=1:1:dim
                    Xnewm(i,j)=Xfood(j)+C3*Flag*Temp*rand*(Xfood(j)-Xm(i,j));%eq.(10)
                end
            end
            for i=1:Nf
                flag_index = floor(2*rand()+1);
                Flag=vec_flag(flag_index);
                for j=1:1:dim
                    Xnewf(i,j)=Xfood(j)+Flag*C3*Temp*rand*(Xfood(j)-Xf(i,j));%eq.(10)
                end
            end
        else %cold
            if rand>0.6 %fight
                for i=1:Nm
                    for j=1:1:dim
                        FM=exp(-(0.5*fitnessBest_f(1)+0.5*fitnessBest_f(2))/(0.5*fitness_m(i,1)+0.5*fitness_m(i,2)+eps));%eq.(13)
                        Xnewm(i,j)=Xm(i,j) +C3*FM*rand*(Q*Xbest_f(j)-Xm(i,j));%eq.(11)
                        
                    end
                end
                for i=1:Nf
                    for j=1:1:dim
                        FF=exp(-(0.5*fitnessBest_m(1)+0.5*fitnessBest_m(2))/(0.5*fitness_f(i,1)+0.5*fitness_f(i,2)+eps));%eq.(14)
                        Xnewf(i,j)=Xf(i,j)+C3*FF*rand*(Q*Xbest_m(j)-Xf(i,j));%eq.(12)
                    end
                end
            else%mating
                for i=1:Nm
                    for j=1:1:dim
                        Mm=exp(-(0.5*fitness_f(i,1)+0.5*fitness_f(i,2))/((0.5*fitness_m(i,1)+0.5*fitness_m(i,2))+eps));%eq.(17)
                        Xnewm(i,j)=Xm(i,j) +C3*rand*Mm*(Q*Xf(i,j)-Xm(i,j));%eq.(15
                    end
                end
                for i=1:Nf
                    for j=1:1:dim
                        Mf=exp(-(0.5*fitness_m(i,1)+0.5*fitness_m(i,2))/((0.5*fitness_f(i,1)+0.5*fitness_f(i,2))+eps));%eq.(18)
                        Xnewf(i,j)=Xf(i,j) +C3*rand*Mf*(Q*Xm(i,j)-Xf(i,j));%eq.(16)
                    end
                end
                flag_index = floor(2*rand()+1);
                egg=vec_flag(flag_index);
                if egg==1;
                    [cs_worst_m,gworst_m] = max(fitness_m(:,2));
                    Xnewm(gworst_m,:)=lb+rand*(ub-lb);%eq.(19)
                    [cs_worst_f,gworst_f] = max(fitness_f(:,2));
                    Xnewf(gworst_f,:)=lb+rand*(ub-lb);%eq.(20)
                end
            end
        end
    end
    
    for j=1:Nm
        Flag4ub=Xnewm(j,:)>ub;
        Flag4lb=Xnewm(j,:)<lb;
        Xnewm(j,:)=(Xnewm(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        y = py.mdn.mdlstm(X_train,X_test,y_train,y_test,Xnewm(j,1),Xnewm(j,2),Xnewm(j,3)).double;
        if dominates(y,fitness_m(j,:))
            fitness_m(j,:)=y;
            Xm(j,:)= Xnewm(j,:);
        end
    end
    [Archive_X_m, Archive_F_m, Archive_member_no_m] = UpdateArchive(Archive_X_m, Archive_F_m, Xm, fitness_m, Archive_member_no_m);
    if Archive_member_no_m>ArchiveMaxSize
        Archive_mem_ranks_m=RankingProcess(Archive_F_m, ArchiveMaxSize, obj_no);
        [Archive_X_m, Archive_F_m, Archive_mem_ranks_m, Archive_member_no_m]=HandleFullArchive(Archive_X_m, Archive_F_m, Archive_member_no_m, Archive_mem_ranks_m, ArchiveMaxSize);
    else
        Archive_mem_ranks_m=RankingProcess(Archive_F_m, ArchiveMaxSize, obj_no);
    end
    Archive_mem_ranks_m=RankingProcess(Archive_F_m, ArchiveMaxSize, obj_no);
    index_m=RouletteWheelSelection(1./Archive_mem_ranks_m);
    if index_m==-1
        index_m=1;
    end
    fitnessBest_m=Archive_F_m(index_m,:);
    Xbest_m=Archive_X_m(index_m,:)';
%     [Ybest1,gbest1] = min(fitness_m);
    for j=1:Nf
        Flag4ub=Xnewf(j,:)>ub;
        Flag4lb=Xnewf(j,:)<lb;
        Xnewf(j,:)=(Xnewf(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        y = py.mdn.mdlstm(X_train,X_test,y_train,y_test,Xnewf(j,1),Xnewf(j,2),Xnewf(j,3)).double;
        if dominates(y,fitness_f(j,:))
            fitness_f(j,:)=y;
            Xf(j,:)= Xnewf(j,:);
        end
    end
    [Archive_X_f, Archive_F_f, Archive_member_no_f]=UpdateArchive(Archive_X_f, Archive_F_f, Xf, fitness_f, Archive_member_no_f);
    if Archive_member_no_f>ArchiveMaxSize
        Archive_mem_ranks_f=RankingProcess(Archive_F_f, ArchiveMaxSize, obj_no);
        [Archive_X_f, Archive_F_f, Archive_mem_ranks_f, Archive_member_no_f]=HandleFullArchive(Archive_X_f, Archive_F_f, Archive_member_no_f, Archive_mem_ranks_f, ArchiveMaxSize);
    else
        Archive_mem_ranks_f=RankingProcess(Archive_F_f, ArchiveMaxSize, obj_no);
    end
    Archive_mem_ranks_f=RankingProcess(Archive_F_f, ArchiveMaxSize, obj_no);
    index_f=RouletteWheelSelection(1./Archive_mem_ranks_f);
    if index_f==-1
        index_f=1;
    end
    fitnessBest_f=Archive_F_f(index_f,:);
    Xbest_f=Archive_X_f(index_f,:)';
    %     [Ybest2,gbest2] = min(fitness_f);
    %
    %     if Ybest1<fitnessBest_m
    %         Xbest_m = Xm(gbest1,:);
    %         fitnessBest_m=Ybest1;
    %     end
    %     if Ybest2<fitnessBest_f
    %         Xbest_f = Xf(gbest2,:);
    %         fitnessBest_f=Ybest2;
    %     end
    X = [Xm;Xf];
    fitness = [fitness_m;fitness_f];
    [Archive_X, Archive_F, Archive_member_no]=UpdateArchive(Archive_X, Archive_F, X, fitness, Archive_member_no);
    if Archive_member_no>ArchiveMaxSize
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
        [Archive_X, Archive_F, Archive_mem_ranks, Archive_member_no]=HandleFullArchive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize);
    else
        Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    end
    Archive_mem_ranks=RankingProcess(Archive_F, ArchiveMaxSize, obj_no);
    index=RouletteWheelSelection(1./Archive_mem_ranks);
    if index==-1
        index=1;
    end
    GYbest=Archive_F(index,:);
    Xfood=Archive_X(index,:)';
%     
%     if Ybest1<Ybest2
%         gbest_t(t)=min(Ybest1);
%     else
%         gbest_t(t)=min(Ybest2);
%     end
%     if fitnessBest_m<fitnessBest_f
%         GYbest=fitnessBest_m;
%         Xfood=Xbest_m;
%     else
%         GYbest=fitnessBest_f;
%         Xfood=Xbest_f;
%     end
    display(['Generation # ' num2str(t)]);
end
fval = GYbest;
end





