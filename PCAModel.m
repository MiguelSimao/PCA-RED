classdef PCAModel
    %PCAMODEL Summary of this class goes here
    %   The purpose of this code is to create a dimensionality reduction
    %   model based on PCA. This is different from other implementations
    %   because it allows to build a reduction model with parameters
    %   calculated with a training data set and use that trained model on
    %   new data.
    
    properties
        dim = []
        coeff = []
        mu = []
    end
    
    methods
        function this = PCAModel(trainingData,dim)
            % data has columns as observations
            trainingData = trainingData';
            
            [N,M] = size(trainingData);

            fprintf('''%i'' data points with dimension ''%i'' found. Training... ',N,M);

            % Calculate parameters:
            [coeff,~,~,~,~,mu] = pca(trainingData,'NumComponents',dim);

            % Save parameters:
            this.coeff = coeff;
            this.mu = mu;
            this.dim = dim;
            fprintf('Done.\n');

        end
        
        function [Y,perform] = run(this,data)
            data = data';
            
            if isempty(this.coeff)
                error('This model is not trained. Not enough inputs.');
            end
            
            X = data;
            N = size(X,1); % number of points

            % Center data:
            score = X - repmat(this.mu,N,1);

            % Rotate data:
            score = score * this.coeff;

            % Reconstruct the data:
            Y = score*this.coeff' + repmat(this.mu,N,1);
            
            % Calculate performance:
            perform = this.L2Norm(data,Y);
            
        end
        
    end
    methods (Static)
        function perform = L2Norm(X,Y)
            % L2Norm: calculate mean L2-norm between X and Y.
            
            % Number of points:
            N = size(X,1);
            
            % Elementwise error:
            E = (X-Y).^2;
            
            tmp = sqrt(sum(E,2));
            
            % Calculate mean:
            perform = 1/N * sum(tmp);
        end
    end
    
end

