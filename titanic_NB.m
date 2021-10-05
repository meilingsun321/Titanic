function [pw,my_m,my_std] = titanic_NB(X_train,y_train)  
[row,~] = size(X_train);  %X_train为7×700矩阵，返回[7,700]，row=7
lab = unique(y_train);  %标签 返回与y_train中相同的数据且不重复有顺序[1 2]
numClass = length(lab);  %类别 numClass=2
for i = 1 : numClass
    pw(i)= sum(y_train==lab(i));
    pw(i) = pw(i)/length(y_train);
end
for i = 1 : row
    for j = 1:numClass
        temp =X_train(i,y_train==lab(j));  % prior
        my_m(i,j) = mean(temp);    % compute the mean
        my_std(i,j)= std(temp);    % compute the std 
    end    
end