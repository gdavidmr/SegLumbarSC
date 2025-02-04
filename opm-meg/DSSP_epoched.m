function [denoised_data] = DSSP_epoched(data, S, num_components)
    [channels, time_points, epochs] = size(data);
    denoised_data = zeros(size(data));
    
    for e = 1:epochs
        epoch_data = data(:,:,e);
        denoised_data(:,:,e) = DSSP(epoch_data, S, num_components);
    end
end
