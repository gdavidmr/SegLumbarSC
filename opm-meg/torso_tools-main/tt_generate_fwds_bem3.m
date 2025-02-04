function L = tt_generate_fwds_bem3(S)
    % Generate the lead fields for a 3-shell BEM of the spinal cord region

    % Ensure required fields are present
    if ~isfield(S, 'pos'); error('Please specify the source positions!'); end
    if ~isfield(S, 'posunits'); error('Please specify the current position units!'); end
    if ~isfield(S, 'ori'); S.ori = []; end  % Allow empty orientations
    if ~isfield(S, 'sensors'); error('Please specify the sensor structure!'); end
    if ~isfield(S, 'ci'); S.ci = [.62 .05 .23]; end  % Conductivity (inner)
    if ~isfield(S, 'co'); S.co = [.23 .23 0]; end   % Conductivity (outer)
    if ~isfield(S, 'isa'); S.isa = []; end  % ISA flag

    % Ensure BEM operators are available
    if isempty(which('hbf_BEMOperatorsPhi_LC'))
        tt_add_bem;  % Load BEM library
    end

    % Load and prepare the BEM meshes 
    meshes = tt_load_meshes([], S.names); 
    [~, sf] = tt_determine_mesh_units(meshes);  % Scaling factor for units
    bmeshes = {};  

    % Visualize the meshes (optional)
    figure; hold on;
    for ii = 1:numel(meshes)
        bmeshes{ii}.p = meshes{ii}.vertices / sf;  % Convert to meters
        bmeshes{ii}.e = meshes{ii}.faces;  % Mesh faces
        plot3(bmeshes{ii}.p(:,1), bmeshes{ii}.p(:,2), bmeshes{ii}.p(:,3), 'm.')
    end

    % Convert sensors into meters
    S.sensors = ft_convert_units(S.sensors, 'm');

    % Prepare the source model (ensure it's in meters)
    cfg = [];
    cfg.method = 'basedongrid';
    cfg.sourcemodel.pos = S.pos;  % Source positions
    cfg.sourcemodel.unit = S.posunits;  % Source unit (ensure it's meters)
    src = ft_prepare_sourcemodel(cfg);
    src = ft_convert_units(src, 'm');  % Convert source model to meters

    % Extract sensor coil positions and orientations
    coils = [];
    coils.p = S.sensors.coilpos;
    coils.n = S.sensors.coilori;

    % Plot sources and sensors (optional visualization)
    plot3(src.pos(:,1), src.pos(:,2), src.pos(:,3), 'g*')  % Source positions
    plot3(coils.p(:,1), coils.p(:,2), coils.p(:,3), 'ko')  % Sensor positions

    % Conductivity settings for the BEM model
    ci = S.ci;  % Inner conductivities (for spinal cord)
    co = S.co;  % Outer conductivities (air or non-conductive medium)

    % Generate the BEM transfer matrix
    D = hbf_BEMOperatorsPhi_LC(bmeshes);  % BEM operators
    if isempty(S.isa)
        % Regular BEM
        Tphi_full = hbf_TM_Phi_LC(D, ci, co);
    else
        % Apply ISA (if specified)
        fprintf('%-40s: %30s\n', 'Applying ISA', S.names{S.isa});
        Tphi_full = hbf_TM_Phi_LC_ISA2(D, ci, co, S.isa);
    end

    % Calculate transfer matrix for magnetic field
    DB = hbf_BEMOperatorsB_Linear(bmeshes, coils);  % Magnetic field operator
    TB = hbf_TM_Bvol_Linear(DB, Tphi_full, ci, co);  % Volume conductor model

    % Compute the lead field matrix (handle orientations)
    if isempty(S.ori)
        % Isotropic sources (no specific orientation)
        L = S.sensors.tra * hbf_LFM_B_LC(bmeshes, coils, TB, src.pos);
    else
        % Directional sources with provided orientations (S.ori)
        L = S.sensors.tra * hbf_LFM_B_LC(bmeshes, coils, TB, S.pos, S.ori);
    end
end
