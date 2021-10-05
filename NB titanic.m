%Naive Bayes for Titanic data classification 

%Load the Titanic data and show it
Train= readtable('titanic\train.csv');
Test = readtable('titanic\test.csv');
head(Train)
head(Test)

%Make the prediction by "sex"
disp(grpstats(Train(:,{'Survived','Sex'}), 'Sex'))
gendermdl = grpstats(Train(:,{'Survived','Sex'}), {'Survived','Sex'});
all_female = (gendermdl.GroupCount('0_male') + gendermdl.GroupCount('1_female'))/(577+314);
disp(all_female)

%Preprocess the NaN values
Train.Fare(Train.Fare == 0) = NaN;      % treat 0 fare as NaN
Test.Fare(Test.Fare == 0) = NaN;        % treat 0 fare as NaN
vars = Train.Properties.VariableNames;  % extract column names
figure
imagesc(ismissing(Train))
set(gca,'XTick', 1:12,'XTickLabel',vars);
xtickangle(-45)
% 用平均年龄代替 NaN 年龄
avgAge = nanmean(Train.Age)             % get average age
Train.Age(isnan(Train.Age)) = avgAge;   % replace NaN with the average
Test.Age(isnan(Test.Age)) = avgAge; 

% 用同一等级舱的平均票价代替 NaN 船票
fare = grpstats(Train(:,{'Pclass','Fare'}),'Pclass');   % get class average
disp(fare)

for i = 1:height(fare) % for each |Pclass|
    % 用 Pclass average 替代缺失的船票值
    Train.Fare(Train.Pclass == i & isnan(Train.Fare)) = fare.mean_Fare(i);
    Test.Fare(Test.Pclass == i & isnan(Test.Fare)) = fare.mean_Fare(i);
end

% Multi Cabins，No Cabin etc.
train_cabins = cellfun(@strsplit, Train.Cabin, 'UniformOutput', false); %strsplit在指定分隔符处拆分字符串或字符向量。以元胞数组形式返回输出值，请指定'UniformOutput',false。
test_cabins = cellfun(@strsplit, Test.Cabin, 'UniformOutput', false);

% count the number of tokens
Train.nCabins = cellfun(@length, train_cabins); %返回船舱位置字符串长度
Test.nCabins = cellfun(@length, test_cabins);

% 处理特殊情况 - 只有一等舱的乘客有多个船舱
Train.nCabins(Train.Pclass ~= 1 & Train.nCabins > 1,:) = 1;
Test.nCabins(Test.Pclass ~= 1 & Test.nCabins > 1,:) = 1;

% if |Cabin| is empty, then |nCabins| should be 0
Train.nCabins(cellfun(@isempty, Train.Cabin)) = 0; %isempty判断是否为空
Test.nCabins(cellfun(@isempty, Test.Cabin)) = 0;

% Embarked is not available
% get most frequent value
disp(grpstats(Train(:,{'Survived','Embarked'}), 'Embarked'))
% apply it to missling value
for i = 1 : 891
    if isempty(Train.Embarked{i})
        Train.Embarked{i}='S';
    end
end

for i = 1 : 418
    if isempty(Test.Embarked{i})
        Test.Embarked{i}='S';
    end
end

% 将数据类型从分类型转换为double型
Train.Embarked = double(cell2mat(Train.Embarked)); %cell2mat将元胞数组转化为普通数组
Test.Embarked = double(cell2mat(Test.Embarked));

% change Sex to tpye "double"
for i = 1 : 891
    if strcmp(Train.Sex{i} ,'male') %比较函数strmp()
        Train.Sex{i}=1;
    else
        Train.Sex{i}=0;
    end
end

for i = 1 : 418
    if strcmp(Test.Sex{i} ,'male')
        Test.Sex{i}=1;
    else
        Test.Sex{i}=0;
    end
end
Train.Sex = cell2mat(Train.Sex);
Test.Sex = cell2mat(Test.Sex);

% Del some columns
Train(:,{'Name','Ticket','Cabin','SibSp','Parch'}) = [];
Test(:,{'Name','Ticket','Cabin','SibSp','Parch'}) = [];

% Age - survived
figure
hist (Train.Age(Train.Survived == 0))   % 死者的年龄柱状图
figure
hist (Train.Age(Train.Survived == 1))   % 幸存者的年龄柱状图
data = Train.Variables;
t = data(:,3:8);
l = data(:,2);
X = t';
X = zscore(X,0,2);
y = (l+1)';
idx = randperm(891);
X_train = X(:,idx(1:700));
y_train = y(idx(1:700));
X_test = X(:,idx(701:891));
y_test = y(idx(701:891));
[pw,my_m,my_std] = titanic_NB(X_train,y_train);
[post_p,test_lab] = titanic_testNB(X_test,pw,my_m,my_std,2,6);
right = y_test == test_lab;
rate = sum(right)/length(y_test);
disp(['NB Accuracy:' num2str(rate*100) '%']);

%测试Test
data1 = Test.Variables;
t1 = data1(:,2:7);
X1 = (t1)';
X1 = zscore(X1,0,2);
[post_p,Test_labs] = titanic_testNB(X1,pw,my_m,my_std,2,6);
PassengerId=[1:1:418]';
Survived=Test_labs(1,:)';
Survived=Survived-1;
T=table(PassengerId,Survived)
writetable(T,'submission.csv')  
type 'submission.csv'