%% day-MOFHO
clc,clear

clear classes
obj = py.importlib.import_module('forecast');
py.importlib.reload(obj);

FS_day = xlsread('FS_day_ahead.xlsx','AQI');
data1 = FS_day(:,1:5);
data2 = FS_day(:,end);
load seed_day
rng(s)
k=rand(1,size(FS_day,1));
[ m, n] = sort(k);

input_train=data1(n(1:531),:);
input_test=data1(n(532:end),:);
output_train=data2(n(1:531),:);
output_test=data2(n(532:end),:);

dim=4;
MaxFes=50;
nPop=10;
lb=[15,15,15,15];
ub=[129,129,129,300];
X_train = input_train(1:398,:);
X_test = input_train(399:end,:);
y_train = output_train(1:398,:);
y_test = output_train(399:end,:);
XY = [X_train,y_train];

% KMIX = py.mdn.Gauss_KMIX(XY);
% double(KMIX)

[Best_Pos, Conv_History] = MOFHO(X_train,X_test,y_train,y_test,dim,lb,ub,MaxFes,nPop);
save FS_AQIday_Best_Pos Best_Pos

%% day-forecast
clc,clear

clear classes
obj = py.importlib.import_module('forecast');
py.importlib.reload(obj);

FS_day = xlsread('FS_day_ahead.xlsx','AQI');
data1 = FS_day(:,1:5);
data2 = FS_day(:,end);
load seed_day
rng(s)
k=rand(1,size(FS_day,1));
[ m, n] = sort(k);

input_train=data1(n(1:531),:);
input_test=data1(n(532:end),:);
output_train=data2(n(1:531),:);
output_test=data2(n(532:end),:);
load FS_AQIday_Best_Pos

Matrix = py.forecast.mdlstm_matrix(input_train,input_test,output_train,output_test,int32(Best_Pos(1)),int32(Best_Pos(2)),int32(Best_Pos(3)),int32(Best_Pos(4)));
M = double(Matrix);
RMSE = round(M(1),4);
MAPE = round(M(2),4);
CRPS = round(M(3),4);
disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS)]);

Values = py.forecast.mdlstm_values(input_train,input_test,output_train,output_test,int32(Best_Pos(1)),int32(Best_Pos(2)),int32(Best_Pos(3)),int32(Best_Pos(4)));
V = cell(Values);
mu = V{1,1}.double';
std = V{1,2}.double';
%优0-50
D1 = normcdf(50,mu,std)-normcdf(0,mu,std);
%良50-100
D2 = normcdf(100,mu,std)-normcdf(50,mu,std);
%轻度污染100-200
D3 = normcdf(200,mu,std)-normcdf(100,mu,std);
%中度污染200-300
D4 = normcdf(300,mu,std)-normcdf(200,mu,std);
%重度污染>300
D5 = normcdf(inf,mu,std)-normcdf(300,mu,std);
D = [D1,D2,D3,D4,D5];
for i = 1:size(D,1)
    [MAX_A,MAX_B] = max(D(i,:));
    DJ_forecast(i,:) = MAX_B;
end
y_test = output_test;
for i = 1:size(y_test,1)
    if y_test(i)>0 & y_test(i)<=50
        DJ_true(i,:) = 1;
    elseif y_test(i)>50 & y_test(i)<=100
        DJ_true(i,:) = 2;
    elseif y_test(i)>100 & y_test(i)<=200
        DJ_true(i,:) = 3;
    elseif y_test(i)>200 & y_test(i)<=300
        DJ_true(i,:) = 4;
    else
        DJ_true(i,:) = 5;
    end
end

for i = 1:size(DJ_true,1)
    if DJ_forecast(i) == DJ_true(i)
        S(i) = 1;
    else
        S(i) = 0;
    end
end
GPA = sum(S)/size(DJ_true,1);