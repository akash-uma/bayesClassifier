function [ params, cv_predError ] = crossvalidate_nb( X, Y, varargin )
%CV_LDA2 Summary of this function goes here
%   Inputs:
%       trainX   - features (nSamples x xDim)
%       trainY   - class labels (nSamples x 1)
%       distType - 'diagGauss' (default),'poisson', 'fullGauss',
%                  'sharedGauss'
%       nFolds   - number of cross validation folds (optional, default is 10)
%
%   Output:
%       params       - model parameters and distType
%                         - classPi, mu, and sigma for the gaussian models
%                         - classPi, lambda for the poisson models
%       cv_predError - prediction error across the n folds
%

    nFolds = 10;
    distType = 'diaggauss';
    useRandomSeed = false;
    minVar = 0.01;
    assignopts(who,varargin);
    
    nSamples = length(Y);
    
    % randomly reorder data points
    if useRandomSeed
        rng(0);
    end 
    rng_idx = randperm(nSamples);
    X = X(rng_idx,:);
    Y = Y(rng_idx);
    fdiv = floor(linspace(1,nSamples+1,nFolds+1));
    
    % train on all data to get params
    [params,~] = train_nb(X,Y,'distType',distType,'minVar',minVar);
    
    cv_predError = zeros(1,nFolds);
    % cross-validation on nFolds
    for cvf=1:nFolds
        testMask = false(1,nSamples);
        testMask(fdiv(cvf):(fdiv(cvf+1)-1)) = true;
        trainMask = ~testMask;
        trainX = X(trainMask,:);
        trainY = Y(trainMask);
        testX = X(testMask,:);
        testY = Y(testMask);
        
        % fit model training data
        [cvf_params,~] = train_nb(trainX,trainY,'distType',distType);
        
        % evaluate model on test data
        test_predY = predict_nb(testX,cvf_params);
        cvf_PE = sum(test_predY~=testY)./length(testY);
        cv_predError(cvf) = cvf_PE;
    end    

end

