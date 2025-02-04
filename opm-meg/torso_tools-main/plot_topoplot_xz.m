function plot_topoplot_xz(ax, positions, leadfield_data, cmap)
    % Interpolate sensor data onto a grid for smoother visualization (x-z plane)
    grid_res = 100;  % Resolution of the grid
    [grid_x, grid_z] = meshgrid(linspace(min(positions(:, 1)), max(positions(:, 1)), grid_res), ...
                                linspace(min(positions(:, 3)), max(positions(:, 3)), grid_res));
    interpolated_values = griddata(positions(:, 1), positions(:, 3), leadfield_data, ...
                                   grid_x, grid_z, 'natural');
    % Plot topoplot using contourf
    contourf(ax, grid_x, grid_z, interpolated_values, 30, 'LineStyle', 'none');
    colormap(ax, cmap);
    caxis(ax, [min(leadfield_data), max(leadfield_data)]);
    colorbar(ax);
    axis(ax, 'equal');
    axis(ax, 'off');
end

% function plot_topoplot_xz(ax, positions, leadfield_data, cmap, clims)
%     % Interpolate sensor data onto a grid for smoother visualization (x-z plane)
%     grid_res = 100;  % Resolution of the grid
%     [grid_x, grid_z] = meshgrid(linspace(min(positions(:, 1)), max(positions(:, 1)), grid_res), ...
%                                 linspace(min(positions(:, 3)), max(positions(:, 3)), grid_res));
%     interpolated_values = griddata(positions(:, 1), positions(:, 3), leadfield_data, ...
%                                    grid_x, grid_z, 'natural');
%     % Plot topoplot using contourf
%     contourf(ax, grid_x, grid_z, interpolated_values, 30, 'LineStyle', 'none');
%     colormap(ax, cmap);
%     caxis(ax, clims);  % Use provided color limits
%     colorbar(ax);
%     axis(ax, 'equal');
%     axis(ax, 'off');
% end