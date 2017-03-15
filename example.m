% Training data set:
X = randn(10,10000);

% Train model with reduction to 3D:
model = PCAModel(X,3);

% Use and test model:
[Y,perform] = model.run(X);
fprintf('Performance on original set: %.2f\n',perform);

% Altered model:
X2 = X + 1;

% The model with parameters calculated on the "training" data set is used on another set:
[Y2,perform2] = model.run(X2);
fprintf('Performance on second set: %.2f\n',perform2);

% To get the reduced data:
Y_reduced = model.reduce(X);