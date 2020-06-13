function [C idx] = mykmeans(X,K)
fprintf('********************************\n');
fprintf('K-means algorithm\n');
rng(1);
n=size(X,2);
%per=randperm(n);
per = [1:n];
C=X(:,per(1:K));
nrms=sum(X.^2);
oldobj=inf;
for ii=1:1000
   % compute idx
   dist=sum(C.^2)' * ones(1,n) + ones(K,1) * nrms - 2*C'*X;
   [tmp idx]=min(dist);
   obj=mean(tmp);
   if mod(ii,10)==1
      fprintf('iter: %d, obj: %d\n',ii,obj);
   end
   for jj=1:K
      ind=find(idx==jj);
      if isempty(ind)
         ran=randi(n);
         C(:,jj)=X(:,ran);
      else
         C(:,jj)=mean(X(:,ind),2);
      end
   end
   if (abs(oldobj-obj)/abs(obj+1e-20) < 1e-6)
      break;
   end
   oldobj=obj;
end
fprintf('********************************\n');
