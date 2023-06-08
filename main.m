%% hour-MOFHO
clc,clear
warning off
clear classes
obj = py.importlib.import_module('mdn');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');

data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);

dim=4;
MaxFes=30;
nPop=10;
lb=[15,15,15,15];
ub=[129,129,129,200];
X_train = input_train(1:0.8*size(input_train,1),:);
X_test = input_train(0.8*size(input_train,1)+1:end,:);
y_train = output_train(1:0.8*size(output_train,1),:);
y_test = output_train(0.8*size(output_train,1)+1:end,:);
XY = [X_train,y_train];

% KMIX = py.mdn.Gauss_KMIX(XY);
% double(KMIX)

%DL2
%CD6
%DL3
%SH3

[Best_Pos, Conv_History] = MOFHO(X_train,X_test,y_train,y_test,dim,lb,ub,MaxFes,nPop);
save DL_AQIhour_Best_Pos Best_Pos

%% hour-forecast
clc,clear

clear classes
obj = py.importlib.import_module('forecast');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);
load DL_AQIhour_Best_Pos

for j = 1:10
    Matrix = py.forecast.mdgru_matrix(input_train,input_test,output_train,output_test,int32(Best_Pos(1)),int32(Best_Pos(2)),int32(Best_Pos(3)),int32(Best_Pos(4)));
    V = cell(Matrix);
    M = V{1,1}.double';
    mu = V{1,2}.double';
    std = V{1,3}.double';
    RMSE = M(1);
    MAPE = M(2);
    CRPS = M(3);

    D1 = normcdf(50,mu,std)-normcdf(0,mu,std);

    D2 = normcdf(100,mu,std)-normcdf(50,mu,std);

    D3 = normcdf(200,mu,std)-normcdf(100,mu,std);

    D4 = normcdf(300,mu,std)-normcdf(200,mu,std);

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
    disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

    r1(j) = RMSE;
    r2(j) = MAPE;
    r3(j) = CRPS;
    r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);

%% MDN
clc,clear

clear classes
obj = py.importlib.import_module('C_mdn');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);

for j = 1:10
    Matrix = py.C_mdn.mdn_matrix(input_train,input_test,output_train,output_test);
    V = cell(Matrix);
    M = V{1,1}.double';
    mu = V{1,2}.double';
    std = V{1,3}.double';
    RMSE = M(1);
    MAPE = M(2);
    CRPS = M(3);

    D1 = normcdf(50,mu,std)-normcdf(0,mu,std);
    D2 = normcdf(100,mu,std)-normcdf(50,mu,std);
    D3 = normcdf(200,mu,std)-normcdf(100,mu,std);
    D4 = normcdf(300,mu,std)-normcdf(200,mu,std);
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

    disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

    r1(j) = RMSE;
    r2(j) = MAPE;
    r3(j) = CRPS;
    r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);

%% MDNGRU
clc,clear

clear classes
obj = py.importlib.import_module('C_mdgru');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);

for j = 1:10
    Matrix = py.C_mdgru.mdlstm_matrix(input_train,input_test,output_train,output_test);
    V = cell(Matrix);
    M = V{1,1}.double';
    mu = V{1,2}.double';
    std = V{1,3}.double';
    RMSE = M(1);
    MAPE = M(2);
    CRPS = M(3);
    D1 = normcdf(50,mu,std)-normcdf(0,mu,std);
    D2 = normcdf(100,mu,std)-normcdf(50,mu,std);
    D3 = normcdf(200,mu,std)-normcdf(100,mu,std);
    D4 = normcdf(300,mu,std)-normcdf(200,mu,std);
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
    disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

    r1(j) = RMSE;
    r2(j) = MAPE;
    r3(j) = CRPS;
    r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);

%% FHO-MDGRU-C1
clc,clear

clear classes
obj = py.importlib.import_module('mdn');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);

dim=4;
MaxFes=20;
nPop=10;
lb=[15,15,15,15];
ub=[129,129,129,200];
X_train = input_train(1:0.8*size(input_train,1),:);
X_test = input_train(0.8*size(input_train,1)+1:end,:);
y_train = output_train(1:0.8*size(output_train,1),:);
y_test = output_train(0.8*size(output_train,1)+1:end,:);
XY = [X_train,y_train];

[Best_Pos, Conv_History] = FHO1(X_train,X_test,y_train,y_test,dim,lb,ub,MaxFes,nPop);
save DL_AQIhour_Best_Pos_C1 Best_Pos

%% hour-forecast
clc,clear

clear classes
obj = py.importlib.import_module('forecast');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);
load DL_AQIhour_Best_Pos_C1

for j = 1:10
    Matrix = py.forecast.mdgru_matrix(input_train,input_test,output_train,output_test,int32(Best_Pos(1)),int32(Best_Pos(2)),int32(Best_Pos(3)),int32(Best_Pos(4)));
    V = cell(Matrix);
    M = V{1,1}.double';
    mu = V{1,2}.double';
    std = V{1,3}.double';
    RMSE = M(1);
    MAPE = M(2);
    CRPS = M(3);

    D1 = normcdf(50,mu,std)-normcdf(0,mu,std);
    D2 = normcdf(100,mu,std)-normcdf(50,mu,std);
    D3 = normcdf(200,mu,std)-normcdf(100,mu,std);
    D4 = normcdf(300,mu,std)-normcdf(200,mu,std);
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
    disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

    r1(j) = RMSE;
    r2(j) = MAPE;
    r3(j) = CRPS;
    r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);

%% FHO-MDGRU-C2
clc,clear

clear classes
obj = py.importlib.import_module('mdn');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);

dim=4;
MaxFes=20;
nPop=10;
lb=[15,15,15,15];
ub=[129,129,129,190];
X_train = input_train(1:0.8*size(input_train,1),:);
X_test = input_train(0.8*size(input_train,1)+1:end,:);
y_train = output_train(1:0.8*size(output_train,1),:);
y_test = output_train(0.8*size(output_train,1)+1:end,:);
XY = [X_train,y_train];

% KMIX = py.mdn.Gauss_KMIX(XY);
% double(KMIX)

[Best_Pos, Conv_History] = FHO2(X_train,X_test,y_train,y_test,dim,lb,ub,MaxFes,nPop);
save DL_AQIhour_Best_Pos_C2 Best_Pos

%% hour-forecast
clc,clear

clear classes
obj = py.importlib.import_module('forecast');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);
load DL_AQIhour_Best_Pos_C2

for j = 1:10
    Matrix = py.forecast.mdgru_matrix(input_train,input_test,output_train,output_test,int32(Best_Pos(1)),int32(Best_Pos(2)),int32(Best_Pos(3)),int32(Best_Pos(4)));
    V = cell(Matrix);
    M = V{1,1}.double';
    mu = V{1,2}.double';
    std = V{1,3}.double';
    RMSE = M(1);
    MAPE = M(2);
    CRPS = M(3);

    D1 = normcdf(50,mu,std)-normcdf(0,mu,std);
    D2 = normcdf(100,mu,std)-normcdf(50,mu,std);
    D3 = normcdf(200,mu,std)-normcdf(100,mu,std);
    D4 = normcdf(300,mu,std)-normcdf(200,mu,std);
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
    disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

    r1(j) = RMSE;
    r2(j) = MAPE;
    r3(j) = CRPS;
    r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);

%% GRU-QR
clc,clear

clear classes
obj = py.importlib.import_module('matrix');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);
input_train =input_train';
output_train=output_train';
input_test=input_test';
output_test=output_test';
[inputn_train,inputps]  =mapminmax(input_train);
[outputn_train,outputps]=mapminmax(output_train);
inputn_test =mapminmax('apply',input_test,inputps); 
outputn_test=mapminmax('apply',output_test,outputps); 
inputSize  = size(inputn_train,1);   
outputSize = size(outputn_train,1);  
numhidden_units=32;
tau=[0.025,0.5,0.975];
for j=1:10
for i = 1:length(tau)
layers = [ ...
    sequenceInputLayer(inputSize)
    gruLayer(numhidden_units) 
    dropoutLayer(0.2)
    fullyConnectedLayer(outputSize)              
    quanRegressionLayer('out',tau(i))];
opts = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu',...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...                
    'LearnRateDropFactor',0.8, ...
    'Verbose',0, ...
    'Plots','training-progress'... 
    );
tic
grunet = trainNetwork(inputn_train ,outputn_train ,layers,opts);
toc;
[grunet,gruoutputr_train]= predictAndUpdateState(grunet,inputn_train);
gruoutput_train = mapminmax('reverse',gruoutputr_train,outputps);
[grunet,gruoutputr_test] = predictAndUpdateState(grunet,inputn_test);
gruoutput_test= mapminmax('reverse',gruoutputr_test,outputps);
sim_gru_qr(:,i)=gruoutput_test';
end

mu = sim_gru_qr(:,2);
std = (sim_gru_qr(:,3)-sim_gru_qr(:,1))/(2*1.959963984540054);

Matrix = py.matrix.mdn_matrix(output_test',mu,mu,std);
M = double(Matrix);
RMSE = round(M(1),4);
MAPE = round(M(2),4);
CRPS = round(M(3),4);

D1 = normcdf(50,mu,std)-normcdf(0,mu,std);
D2 = normcdf(100,mu,std)-normcdf(50,mu,std);
D3 = normcdf(200,mu,std)-normcdf(100,mu,std);
D4 = normcdf(300,mu,std)-normcdf(200,mu,std);
D5 = normcdf(inf,mu,std)-normcdf(300,mu,std);
D = [D1,D2,D3,D4,D5];
for i = 1:size(D,1)
    [MAX_A,MAX_B] = max(D(i,:));
    DJ_forecast(i,:) = MAX_B;
end

y_test = output_test';
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
disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

r1(j) = RMSE;
r2(j) = MAPE;
r3(j) = CRPS;
r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);


%% GRU-KDE
clc,clear

clear classes
obj = py.importlib.import_module('matrix');
py.importlib.reload(obj);

DL_hour = xlsread('DL_hour_ahead.xlsx','AQI');
data1 = DL_hour(:,1:end-1);
data2 = DL_hour(:,end);
load seed_hour
rng(s)
k=rand(1,size(DL_hour,1));
[ m, n] = sort(k);

input_train=data1(n(1:0.8*size(DL_hour,1)),:);
input_test=data1(n(0.8*size(DL_hour,1)+1:end),:);
output_train=data2(n(1:0.8*size(DL_hour,1)),:);
output_test=data2(n(0.8*size(DL_hour,1)+1:end),:);
input_train =input_train';
output_train=output_train';
input_test=input_test';
output_test=output_test';
[inputn_train,inputps]  =mapminmax(input_train);
[outputn_train,outputps]=mapminmax(output_train);
inputn_test =mapminmax('apply',input_test,inputps); 
outputn_test=mapminmax('apply',output_test,outputps); 
inputSize  = size(inputn_train,1);   
outputSize = size(outputn_train,1);  
numhidden_units=32;
for j=1:10
layers = [ ...
    sequenceInputLayer(inputSize)
    gruLayer(numhidden_units) 
    dropoutLayer(0.2)
    fullyConnectedLayer(outputSize)              
    regressionLayer];
opts = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'GradientThreshold',1,...
    'ExecutionEnvironment','cpu',...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',100, ...                
    'LearnRateDropFactor',0.8, ...
    'Verbose',0, ...
    'Plots','training-progress'... 
    );
tic
grunet = trainNetwork(inputn_train ,outputn_train ,layers,opts);
toc;
[grunet,gruoutputr_train]= predictAndUpdateState(grunet,inputn_train);
gruoutput_train = mapminmax('reverse',gruoutputr_train,outputps);
[grunet,gruoutputr_test] = predictAndUpdateState(grunet,inputn_test);
gruoutput_test= mapminmax('reverse',gruoutputr_test,outputps);
sim_gru_kde= gruoutput_test';

e = (output_test'-sim_gru_kde)./sim_gru_kde;
X1=e;
[f,X]=ecdf(X1);
n=length(X1);
h=1.06*std(X1)*n^(-0.2);
syms t
f0=(1/(sqrt(2*pi)*n*h))*sum(exp((-1/2)*((t-X1)./h).^2));%Gaussion
syms x
Alpha = 0.05;
la = solve(int(f0,-inf,x)==Alpha/2,x);
ua = solve(int(f0,-inf,x)==1-Alpha/2,x);
lb=sim_gru_kde.*(1+double(la));
ub=sim_gru_kde.*(1+double(ua));

mu = sim_gru_kde;
std1 = (ub-lb)/(2*1.959963984540054);

Matrix = py.matrix.mdn_matrix(output_test',mu,mu,std1);
M = double(Matrix);
RMSE = round(M(1),4);
MAPE = round(M(2),4);
CRPS = round(M(3),4);

D1 = normcdf(50,mu,std1)-normcdf(0,mu,std1);
D2 = normcdf(100,mu,std1)-normcdf(50,mu,std1);
D3 = normcdf(200,mu,std1)-normcdf(100,mu,std1);
D4 = normcdf(300,mu,std1)-normcdf(200,mu,std1);
D5 = normcdf(inf,mu,std1)-normcdf(300,mu,std1);
D = [D1,D2,D3,D4,D5];
for i = 1:size(D,1)
    [MAX_A,MAX_B] = max(D(i,:));
    DJ_forecast(i,:) = MAX_B;
end

y_test = output_test';
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
disp(['RMSE:',num2str(RMSE),'    MAPE:',num2str(MAPE),'    CRPS:',num2str(CRPS),'    GPA:',num2str(GPA)]);

r1(j) = RMSE;
r2(j) = MAPE;
r3(j) = CRPS;
r4(j) = GPA;
end
disp(['RMSE:         ',num2str(mean(r1)),'           ',num2str(max((max(r1)-mean(r1)),(mean(r1)-min(r1))))]);
disp(['MAPE:         ',num2str(mean(r2)),'           ',num2str(max((max(r2)-mean(r2)),(mean(r2)-min(r2))))]);
disp(['CRPS:         ',num2str(mean(r3)),'           ',num2str(max((max(r3)-mean(r3)),(mean(r3)-min(r3))))]);
disp(['GPA:          ',num2str(mean(r4)),'           ',num2str(max((max(r4)-mean(r4)),(mean(r4)-min(r4))))]);
