# Sliding window statistics
import torch
from prodict import Prodict
import sys
import os
import pickle


class MuSigmaTensor(torch.Tensor):
    # Implements Mu and Sigma sliding window aggregations
    # To be used when testing normalizations using Mu and Sigma
    # groupby: 'channel', 'all'
    # reduceto: 'point', 'tensor'
    def __new__(cls, x, groupby, reduceto, decay='none', decay_factor=5, window_length=5, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, x, groupby, reduceto, decay='none', decay_factor=5, window_length=5):
        super().__init__()
        # x is expected to have the following shape: [batchsize, channels, H, W]
        assert len(x.shape) == 4 # x is a batch of 3D tensors
        assert groupby in ['channel', 'all']
        assert reduceto in ['point', 'tensor']
        assert decay in ['none', 'exp']
        if self.shape[0] > window_length:
            excp = f"Cannot initialize tensor with shape ({self.shape[0]}) larger than the window_length ({window_length})!"
            raise Exception(excp)
        self.groupby = groupby
        self.reduceto = reduceto
        self.decay = decay
        self.decay_factor = decay_factor
        self.window_length = window_length
        self.running_mean, self.running_std = self.ComputeMuSigma()
        
    def weighted_mean(self, x, weights, redact=0):
        if self.groupby == 'channel':
            if self.reduceto == 'point':
                nbe = x.shape[1]*x.shape[2]
                return torch.sum(torch.mul(x, weights))/(torch.sum(weights)*nbe-redact)
            elif self.reduceto == 'tensor':
                return torch.sum(torch.mul(x, weights), axis=0)/(torch.sum(weights)-redact)
        elif self.groupby == 'all':
            if self.reduceto == 'point':
                nbe = x.shape[1]*x.shape[2]*x.shape[3]
                return torch.sum(torch.mul(x, weights))/(torch.sum(weights)*nbe-redact)
            elif self.reduceto == 'tensor':
                return torch.sum(torch.mul(x.view(-1, x.shape[2], x.shape[3]), weights.repeat_interleave(4).view(-1, 1, 1)), axis=0)/(torch.sum(weights)-redact)    
    
    def weighted_std(self, x, mean, weights):
        return torch.sqrt(self.weighted_mean((x-mean)**2, weights=weights, redact=1))

    def ComputeMuSigma(self):
        if self.decay == 'none':
            self.decay_weights = torch.ones(self.shape[0], device=self.device)
        elif self.decay == 'exp':
            self.decay_weights = torch.exp(self.decay_factor*torch.linspace(0, -1, self.shape[0], device=self.device))
        
        if self.groupby == 'channel':
            if self.reduceto == 'point':
                # groupby channel, reduceto point
                means = [self.weighted_mean(x=self[:, c, :, :], weights=self.decay_weights.view(-1, 1, 1)).unsqueeze(0) for c in range(self.shape[1])]
                stds = [self.weighted_std(x=self[:, c, :, :], mean=means[c], weights=self.decay_weights.view(-1, 1, 1)).unsqueeze(0) for c in range(self.shape[1])]
                return torch.cat(means, axis=0).unsqueeze(0), torch.cat(stds, axis=0).unsqueeze(0)
            elif self.reduceto == 'tensor':
                # groupby channel, reduceto tensor
                means = [self.weighted_mean(x=self[:, c, :, :], weights=self.decay_weights.view(-1, 1, 1)).unsqueeze(0) for c in range(self.shape[1])]
                stds = [self.weighted_std(x=self[:, c, :, :], mean=means[c], weights=self.decay_weights.view(-1, 1, 1)).unsqueeze(0) for c in range(self.shape[1])]
                return torch.cat(means, axis=0).unsqueeze(0), torch.cat(stds, axis=0).unsqueeze(0)
        elif self.groupby == 'all':
            if self.reduceto == 'point':
                # groupby all, reduceto point
                mean = self.weighted_mean(self, weights=self.decay_weights.view(-1, 1, 1, 1))
                std = self.weighted_std(self, mean=mean, weights=self.decay_weights.view(-1, 1, 1, 1))
                return mean.unsqueeze(0), std.unsqueeze(0)
            elif self.reduceto == 'tensor':
                # groupby all, reduceto tensor
                mean = self.weighted_mean(self, weights=self.decay_weights.view(-1, 1, 1, 1))
                std = self.weighted_std(self, mean=mean, weights=self.decay_weights.view(-1, 1, 1, 1))
                return mean.unsqueeze(0), std.unsqueeze(0)
            
    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        self.running_mean, self.running_std = self.ComputeMuSigma()
        
    def append(self, item):
        assert len(item.shape) == 4 # append a batch of 3D tensors
        return MuSigmaTensor(torch.cat((item, self), axis=0), 
                             groupby=self.groupby,
                             reduceto=self.reduceto,
                             decay=self.decay,
                             decay_factor=self.decay_factor,
                             window_length=self.window_length)

    def pop_push(self, item):
        # pops the oldest item in the list
        # and pushes the new item
        # (while doing the necessary updates)
        assert len(item.shape) == 4 # pop_push a batch of 3D tensors
        if self.shape[0]+item.shape[0] > self.window_length:
            if self.shape[0] == self.window_length:
                # pop_push
                pop_len = item.shape[0]
                return MuSigmaTensor(torch.cat((item, self[:-pop_len]), axis=0), 
                                     groupby=self.groupby, 
                                     reduceto=self.reduceto,
                                     decay=self.decay,
                                     decay_factor=self.decay_factor,
                                     window_length=self.window_length)
            else:
                pop_len = self.shape[0] + item.shape[0] - self.window_length
                append_len = self.window_length - self.shape[0]
                # append some of the elements
                intermediate_self = self.append(item[:append_len])
                # pop_push the rest
                return MuSigmaTensor(torch.cat((item[append_len:], intermediate_self[:-pop_len]), axis=0), 
                                     groupby=self.groupby, 
                                     reduceto=self.reduceto,
                                     decay=self.decay,
                                     decay_factor=self.decay_factor,
                                     window_length=self.window_length)
        else:
            # append
            return self.append(item)

class MinMaxTensor(torch.Tensor):
    # Implements Min and Max sliding window aggregations
    # To be used when testing normalizations using Min and Max
    def __new__(cls, x, groupby, reduceto, window_length=5, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    def __init__(self, x, groupby, reduceto, window_length=5):
        super().__init__()
        # x is expected to have the following shape: [batchsize, channels, H, W]
        assert len(x.shape) == 4 # x is a batch of 3D tensors
        assert groupby in ['channel', 'all']
        assert reduceto in ['point', 'tensor']
        self.groupby = groupby
        self.reduceto = reduceto
        self.window_length = window_length
        self.running_min, self.running_max = self.ComputeMinMax()
        
    def ComputeMinMax(self):
        if self.groupby == 'channel':
            if self.reduceto == 'point':
                # groupby channel, reduceto point
                mins = [torch.min(self[:, c, :, :]).unsqueeze(0) for c in range(self.shape[1])]
                maxs = [torch.max(self[:, c, :, :]).unsqueeze(0) for c in range(self.shape[1])]
                return torch.cat(mins, axis=0).unsqueeze(0), torch.cat(maxs, axis=0).unsqueeze(0)
            elif self.reduceto == 'tensor':
                # groupby channel, reduceto tensor
                mins = [torch.min(self[:, c, :, :], axis=0).values.unsqueeze(0) for c in range(self.shape[1])]
                maxs = [torch.max(self[:, c, :, :], axis=0).values.unsqueeze(0) for c in range(self.shape[1])]
                return torch.cat(mins, axis=0).unsqueeze(0), torch.cat(maxs, axis=0).unsqueeze(0)
        elif self.groupby == 'all':
            if self.reduceto == 'point':
                # groupby all, reduceto point
                cmin = torch.min(self)
                cmax = torch.max(self)
                return cmin.unsqueeze(0), cmax.unsqueeze(0)
            elif self.reduceto == 'tensor':
                # groupby all, reduceto tensor
                cmin = torch.min(self, axis=0)
                cmax = torch.max(self, axis=0)
                return cmin.values.unsqueeze(0), cmax.values.unsqueeze(0)
            
    def __setitem__(self, index, item):
        super().__setitem__(index, item)
        self.running_min, self.running_max = self.ComputeMinMax()
    
    def append(self, item):
        assert len(item.shape) == 4 # append a batch of 3D tensors
        return MinMaxTensor(torch.cat((item, self), axis=0), 
                            groupby=self.groupby,
                            reduceto=self.reduceto,
                            window_length=self.window_length)
    
    def pop_push(self, item):
        # pops the oldest item in the list
        # and pushes the new item
        # (while doing the necessary updates)
        assert len(item.shape) == 4 # pop_push a batch of 3D tensors
        if self.shape[0]+item.shape[0] > self.window_length:
            if self.shape[0] == self.window_length:
                # pop_push
                pop_len = item.shape[0]
                return MinMaxTensor(torch.cat((item, self[:-pop_len]), axis=0), 
                                    groupby=self.groupby, 
                                    reduceto=self.reduceto,
                                    window_length=self.window_length)
                
            else:
                pop_len = self.shape[0] + item.shape[0] - self.window_length
                append_len = self.window_length - self.shape[0]
                # append some of the elements
                intermediate_self = self.append(item[:append_len])
                # pop_push the rest
                return MinMaxTensor(torch.cat((item[append_len:], intermediate_self[:-pop_len]), axis=0), 
                                    groupby=self.groupby, 
                                    reduceto=self.reduceto,
                                    window_length=self.window_length)
        else:
            # append
            return self.append(item)
        
class ScaledTanh():
    def __init__(self, scale=1):
        self.scale = scale
        self.tanh = torch.nn.Tanh()
    
    def apply_norm(self, x):
        return self.tanh(self.scale * x)
    
    def deapply_norm(self, x):
        return (1/self.scale) * torch.atanh(x)
    
    def append(self, item):
        return self
    
    def pop_push(self, item):
        return self

class MuSigmaTanh(MuSigmaTensor):
    def __new__(cls, x, groupby, reduceto, decay='none', decay_factor=5, window_length=5, scale=1, *args, **kwargs):
        return super().__new__(cls, x, groupby, reduceto, decay, decay_factor, window_length, *args, **kwargs)
        
    def __init__(self, x, groupby, reduceto, decay='none', decay_factor=5, window_length=5, scale=1):
        super().__init__(x, groupby, reduceto, decay, decay_factor, window_length)
        self.scale = scale
        self.scaledtanh = ScaledTanh(scale)

    def append(self, item):
        assert len(item.shape) == 4 # append a batch of 3D tensors
        return MuSigmaTanh(torch.cat((item, self), axis=0), 
                             groupby=self.groupby,
                             reduceto=self.reduceto,
                             decay=self.decay,
                             decay_factor=self.decay_factor,
                             scale=self.scale,
                             window_length=self.window_length)

    def pop_push(self, item):
        # pops the oldest item in the list
        # and pushes the new item
        # (while doing the necessary updates)
        assert len(item.shape) == 4 # pop_push a batch of 3D tensors
        if self.shape[0]+item.shape[0] > self.window_length:
            if self.shape[0] == self.window_length:
                # pop_push
                pop_len = item.shape[0]
                return MuSigmaTanh(torch.cat((item, self[:-pop_len]), axis=0), 
                                     groupby=self.groupby, 
                                     reduceto=self.reduceto,
                                     decay=self.decay,
                                     decay_factor=self.decay_factor,
                                     scale=self.scale,
                                     window_length=self.window_length)
            else:
                pop_len = self.shape[0] + item.shape[0] - self.window_length
                append_len = self.window_length - self.shape[0]
                # append some of the elements
                intermediate_self = self.append(item[:append_len])
                # pop_push the rest
                return MuSigmaTanh(torch.cat((item[append_len:], intermediate_self[:-pop_len]), axis=0), 
                                     groupby=self.groupby, 
                                     reduceto=self.reduceto,
                                     decay=self.decay,
                                     decay_factor=self.decay_factor,
                                     scale=self.scale,
                                     window_length=self.window_length)
        else:
            # append
            return self.append(item)



def newAggTensor(x, norm_type, groupby='channel', reduceto='tensor', decay='none', decay_factor=5, window_length=5, tanh_scale=1):
    # This method is to be called 'universally' 
    # so the aggregation type can be precised in config
    # e.g. agg_tensor = newAggTensor(some_tensors, norm_type=config.agg_type)
    if (norm_type == 'MuSigma'):
        return MuSigmaTensor(x, 
                             groupby=groupby, 
                             reduceto=reduceto, 
                             decay=decay, 
                             decay_factor=decay_factor, 
                             window_length=window_length).to(x.device)
    elif (norm_type == 'MinMax'):
        return MinMaxTensor(x, 
                            groupby=groupby, 
                            reduceto=reduceto, 
                            window_length=window_length).to(x.device)
    elif (norm_type == 'Tanh'):
        # In this case, x is None
        return ScaledTanh(tanh_scale)
    elif (norm_type == 'MuSigmaTanh'):
        return MuSigmaTanh(x, 
                           groupby=groupby, 
                           reduceto=reduceto, 
                           decay=decay, 
                           decay_factor=decay_factor, 
                           window_length=window_length,
                           scale=tanh_scale).to(x.device)
    else:
        raise Exception(f'Normalization type {norm_type} unknown.')
    
def agg_normalize(x, agg_tensor, sigma_scale=1):
    # 'Universal' method to normalize tensors
    # e.g. x_normalized = normalize(x, some_AggTensor)
    if type(agg_tensor) == MuSigmaTensor:
        running_mean = agg_tensor.running_mean
        running_std = agg_tensor.running_std
        
        # shape adjustment
        if agg_tensor.groupby == 'channel' and agg_tensor.reduceto == 'point':
            running_mean = running_mean.view(1, -1, 1, 1)
            running_std = running_std.view(1, -1, 1, 1)
            
        return torch.div(x-running_mean, sigma_scale*running_std)
    elif type(agg_tensor) == MinMaxTensor:
        running_min = agg_tensor.running_min
        running_max = agg_tensor.running_max
        
        # shape adjustment
        if agg_tensor.groupby == 'channel' and agg_tensor.reduceto == 'point':
            running_min = running_min.view(1, -1, 1, 1)
            running_max = running_max.view(1, -1, 1, 1)
        
        return (torch.div(x-running_min, running_max-running_min)*2)-1
    elif type(agg_tensor) == ScaledTanh:
        return agg_tensor.apply_norm(x)
    elif type(agg_tensor) == MuSigmaTanh:
        running_mean = agg_tensor.running_mean
        running_std = agg_tensor.running_std
        
        # shape adjustment
        if agg_tensor.groupby == 'channel' and agg_tensor.reduceto == 'point':
            running_mean = running_mean.view(1, -1, 1, 1)
            running_std = running_std.view(1, -1, 1, 1)
        msn = torch.div(x-running_mean, sigma_scale*running_std)
        return agg_tensor.scaledtanh.apply_norm(msn)
    else:
        raise Exception("Unknown aggregation type.")
    
def agg_denormalize(x, agg_tensor, sigma_scale=1, eps=1e-4):
    # 'Universal' method to denormalize tensors
    # e.g. x = denormalize(x_normalized, some_AggTensor)
    if type(agg_tensor) == MuSigmaTensor:
        running_mean = agg_tensor.running_mean
        running_std = agg_tensor.running_std
        
        # shape adjustment
        if agg_tensor.groupby == 'channel' and agg_tensor.reduceto == 'point':
            running_mean = running_mean.view(1, -1, 1, 1)
            running_std = running_std.view(1, -1, 1, 1)
        
        return torch.mul(x, sigma_scale*running_std) + running_mean
    elif type(agg_tensor) == MinMaxTensor:
        running_min = agg_tensor.running_min
        running_max = agg_tensor.running_max
        
        # shape adjustment
        if agg_tensor.groupby == 'channel' and agg_tensor.reduceto == 'point':
            running_min = running_min.view(1, -1, 1, 1)
            running_max = running_max.view(1, -1, 1, 1)
            
        return torch.mul((x+1)/2, running_max-running_min) + running_min
    elif type(agg_tensor) == ScaledTanh:
        return agg_tensor.deapply_norm(x.clip(-1 + eps, 1 - eps))
    elif type(agg_tensor) == MuSigmaTanh:
        running_mean = agg_tensor.running_mean
        running_std = agg_tensor.running_std
        
        # shape adjustment
        if agg_tensor.groupby == 'channel' and agg_tensor.reduceto == 'point':
            running_mean = running_mean.view(1, -1, 1, 1)
            running_std = running_std.view(1, -1, 1, 1)
        msn = agg_tensor.scaledtanh.deapply_norm(x.clip(-1 + eps, 1 - eps))
        return torch.mul(msn, sigma_scale*running_std) + running_mean
    else:
        raise Exception("Unknown aggregation type.")
    
class WindowNormalizer():
    def __init__(self, normalization_args):
        self.window_length = normalization_args.window_length
        self.type = normalization_args.type
        self.sigma_scale = normalization_args.sigma_scale
        self.tanh_scale = normalization_args.tanh_scale
        self.groupby = normalization_args.groupby
        self.decay = normalization_args.decay
        self.decay_factor = normalization_args.decay_factor
        self.eps = normalization_args.eps
        self.reduceto = normalization_args.reduceto
        
        if self.type == 'Tanh': 
            self.agg_tensor = ScaledTanh(self.tanh_scale)
        else : 
            self.agg_tensor = None

        
    def pop_push(self, x):
        if self.agg_tensor is None:
            self.agg_tensor = newAggTensor(x, 
                                    norm_type=self.type, 
                                    groupby=self.groupby, 
                                    reduceto=self.reduceto, 
                                    decay=self.decay, 
                                    decay_factor=self.decay_factor, 
                                    window_length=self.window_length, 
                                    tanh_scale=self.tanh_scale)
        else : 
            self.agg_tensor = self.agg_tensor.pop_push(x)

        return self
    
    def normalize(self, x):
        res = agg_normalize(x, self.agg_tensor, sigma_scale=self.sigma_scale)
        assert res.ndimension() == x.ndimension()
        return res
    
    def denormalize(self, x):
        res = agg_denormalize(x, self.agg_tensor, sigma_scale=self.sigma_scale, eps=self.eps)
        assert res.ndimension() == x.ndimension()
        return res
    
    def is_full(self):
        if self.agg_tensor is None:
            return False
        elif self.type == 'Tanh':
            return True
        else:
            return self.agg_tensor.shape[0] == self.window_length

