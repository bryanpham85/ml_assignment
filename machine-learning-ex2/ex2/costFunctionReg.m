function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

noOfFeature = size(X,1); % get the number of feature or x(i)

tempJ = 0;

% calculate cost for logistic regression
for i=1:m
  tempJ += (-y(i)*log(sigmoid(theta' * X(i,:)'))) - ((1-y(i))*log(1-sigmoid(theta' * X(i,:)')));
end

%calculate the regulization
regularization = 0;
tempReg = 0;

for i=2:size(theta)
  tempReg += theta(i)^2;
end
regularization = tempReg * lambda / (2*m);


J = 1/m*tempJ +regularization;


%Calculate for grad at j = 0
tempGradAtZero = 0;
for j=1:m
  tempGradAtZero += (sigmoid(theta' * X(j,:)') - y(j))*X(j,1); %because x_0 is at index 1 in octave
end

gradAtZero = 1/m * tempGradAtZero;
grad(1) = gradAtZero;

% calculate the grad for  j >0 j=1 means 2 in context index of octave
for i=2:size(theta)
   tempGrad = 0;
   for j=1:m
      tempGrad += (sigmoid(theta' * X(j,:)') - y(j))*X(j,i);
   end
   grad(i) = 1/m*tempGrad + lambda/m*theta(i);
end





% =============================================================

end
