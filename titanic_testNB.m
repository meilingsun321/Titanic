function [post_p,test_lab] = titanic_testNB(xt,pw,my_mean,my_std,numClass,numVar) %（X_test,pw,my_m,my_std,2,6）
[~,len] = size(xt);  %len=191
for k = 1 : len
    temp = xt(:,k);
    for i = 1: numClass
        prod = 1;
        for j = 1:numVar
            prod = prod*mvnpdf(temp(j),my_mean(j,i),my_std(j,i));  %MU是均值，SIGMA是方差。y是X在以MU为均值、SIGMA为方差的正态分布下的概率。
%            prod = prod*pdf('normal',temp(j),my_mean(j,i),my_std(j,i)); 
        end
        post_p(k,i) = prod*pw(i);
    end
    [~,inx] = max(post_p(k,:));
    test_lab(k) = inx;
end