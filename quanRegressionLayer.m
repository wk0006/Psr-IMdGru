classdef quanRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    properties
        tau
    end
    methods
        function layer = quanRegressionLayer(name,tau)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
           % global tau
            % Set layer name.
            layer.tau = tau;
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'quantile error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
           % global tau
            % Calculate MAE.
  
            R = size(Y,1);
            quantileError = sum(max(layer.tau*(T-Y),(1-layer.tau)*(Y-T)))/R;  %有部分点不可导，可以改进
            
            % Take mean over mini-batch.
            N = size(Y,3);
            loss = sum(quantileError)/N;
        end
        function dLdY = backwardLoss(layer,Y,T)
           
            dLdY =  single(-layer.tau*(T-Y>= 0) + (1-layer.tau) * (Y-T>=0));
                    
        end
        
        
    end
end


