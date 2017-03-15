% Training data set:
X = randn(10,10000);

% Train model with reduction to 3D:
model = PCAModel(X,3);

% Use and test model:
[Y,perform] = model.run(X);
fprintf('Performance on original set: %.2f\n',perform);

X2 = X + 1;

[Y2,perform2] = model.run(X2);
fprintf('Performance on second set: %.2f\n',perform2);