function [ predictions, diff_logP ] = predict_nb( testX, params )
%
%   Input:
%       testX - features (nSamples x xDim)
%       params - trained parameters
%
%   Output:
%       predictions - (nSamples x 1)
%

    nClasses = params.nClasses;
    distType = params.distType;
    classLabels = params.classLabels;
    
    nSamples = size(testX,1);
    logP = zeros(nSamples,nClasses);
    
    if strcmpi(distType,'poisson')
        for ii=1:nClasses
            Lambda = params.Lambda(ii,:);
            classPi = params.classPi(ii);
            logP(:,ii) = testX*log(Lambda') - sum(Lambda) + log(classPi);
        end
    else % gaussian
        for ii=1:nClasses
            Mu = params.Mu(ii,:);
            Sigma = squeeze(params.Sigma(ii,:,:));
            classPi = params.classPi(ii);
            centeredX = bsxfun(@minus,testX,Mu);
            logP(:,ii) = -1/2 .* (diag(centeredX/Sigma*centeredX') + log(det(Sigma))) + log(classPi);
        end
    end
    
    [~, maxIdx] = max(logP,[],2);
    predictions = classLabels(maxIdx);
    diff_logP = logP(:,2)-logP(:,1);
    
end

