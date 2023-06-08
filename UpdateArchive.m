function [Archive_X_updated, Archive_F_updated, Archive_member_no]=UpdateArchive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no)
Archive_X_temp=[Archive_X ; Particles_X];
Archive_F_temp=[Archive_F ; Particles_F];
o=zeros(1,size(Archive_F_temp,1));
for i=1:size(Archive_F_temp,1)
    o(i)=0;
    for j=1:i-1
        if any(Archive_F_temp(i,:) ~= Archive_F_temp(j,:))
            if dominates(Archive_F_temp(i,:),Archive_F_temp(j,:))
                o(j)=1;
            elseif dominates(Archive_F_temp(j,:),Archive_F_temp(i,:))
                o(i)=1;
                break;
            end
        else
            o(j)=1;
            o(i)=1;
        end
    end
end
Archive_member_no=0;
index=0;
if sum(o)~=0
    Np = size(Archive_F_temp,1);
    DOMINATED = zeros(Np,1);
    all_perm = nchoosek(1:Np,2);    % Possible permutations
    all_perm = [all_perm; [all_perm(:,2) all_perm(:,1)]];
    d = dominates1(Archive_F_temp(all_perm(:,1),:),Archive_F_temp(all_perm(:,2),:));
    dominated_particles = unique(all_perm(d==1,2));
    DOMINATED(dominated_particles) = 1;
    Archive_X_updated  = Archive_X_temp(~DOMINATED,:);
    Archive_F_updated = Archive_F_temp(~DOMINATED,:);
    Archive_member_no = size(Archive_X_updated);
else
    for i=1:size(Archive_X_temp,1)
    if o(i)==0
        Archive_member_no=Archive_member_no+1;
        Archive_X_updated(Archive_member_no,:)=Archive_X_temp(i,:);
        Archive_F_updated(Archive_member_no,:)=Archive_F_temp(i,:);
    else
        index=index+1;
    end
    end
end
end
