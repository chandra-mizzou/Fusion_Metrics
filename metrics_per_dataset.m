% Present running code 
clear all;
close all;
clc;

% Set folder paths
folder1 = '/MATLAB Drive/dataset_21/M3FD_21/vi';
folder2 = '/MATLAB Drive/dataset_21/M3FD_21/ir';
folder3 = '/MATLAB Drive/imagepairs_21_test/Oursv2_21/M3FD';

fid = fopen('/MATLAB Drive/csv_21/M3FD/Oursv2_MF3D_all_metrics.csv', 'w');


% Gather both PNG and JPG files from each folder (add JPG if needed)
files1 = dir(fullfile(folder1, '*.png')); 
files2 = dir(fullfile(folder2, '*.png'));
files3 = dir(fullfile(folder3, '*.png')); 
%files3 = dir(fullfile(folder3, '*.jpg'));

% Sort files alphabetically to ensure correct pairing
[~, idx1] = sort({files1.name}); files1 = files1(idx1);
[~, idx2] = sort({files2.name}); files2 = files2(idx2);
[~, idx3] = sort({files3.name}); files3 = files3(idx3);

numImages = min([length(files1), length(files2), length(files3)]);
numMetrics = 13; % Number of metrics you calculate

results = zeros(numImages, numMetrics);

for i = 1:numImages
    img1 = imread(fullfile(folder1, files1(i).name));
    img2 = imread(fullfile(folder2, files2(i).name));
    fused = imread(fullfile(folder3, files3(i).name));
    
    % Convert to grayscale if images are RGB
    if size(img1,3) == 3, img1 = rgb2gray(img1); end
    if size(img2,3) == 3, img2 = rgb2gray(img2); end
    if size(fused,3) == 3, fused = rgb2gray(fused); end
    
    % Resize if necessary to match dimensions
    if ~isequal(size(img1), size(fused))
        img1 = imresize(img1, size(fused));
    end
    if ~isequal(size(img2), size(fused))
        img2 = imresize(img2, size(fused));
    end
    
    % Calculate metrics and store in results matrix
    results(i,1)  = metricsVariance(img1,img2,fused);
    results(i,2)  = metricsSsim(img1,img2,fused);
    results(i,3)  = metricsSpatial_frequency(img1,img2,fused);
    results(i,4)  = metricsRmse(img1,img2,fused);
    results(i,5)  = metricsQcv(img1,img2,fused);
    results(i,6)  = metricsQcb(img1,img2,fused);
    results(i,7)  = metricsQabf(img1,img2,fused);
    results(i,8)  = metricsPsnr(img1,img2,fused);
    results(i,9)  = metricsMutinf(img1,img2,fused);
    results(i,10) = metricsEntropy(img1,img2,fused);
    results(i,11) = metricsEdge_intensity(img1,img2,fused);
    results(i,12) = metricsCross_entropy(img1,img2,fused);
    results(i,13) = metricsAvg_gradient(img1,img2,fused);
end

% Save all metrics to text file with column headers
% metric_names = {'Variance','SSIM','SF','RMSE','Qcv','Qcb','Qabf','PSNR','MI','Entropy','EI','CE','AG'};
% fid = fopen('BDLFusion_M3FD_all_metrics.txt', 'w');
% fprintf(fid, 'S.No.');
% for k = 1:numMetrics
%     fprintf(fid, '\t%s', metric_names{k});
% end
% fprintf(fid, '\n');
% for i = 1:numImages
%     fprintf(fid, '%d', i);
%     for k = 1:numMetrics
%         fprintf(fid, '\t%.6f', results(i,k));
%     end
%     fprintf(fid, '\n');
% end
% fclose(fid);

metric_names = {'Variance', 'SSIM', 'SF', 'RMSE', 'Qcv', 'Qcb', 'Qabf', 'PSNR', 'MI', 'Entropy', 'EI', 'CE', 'AG'};

% Example results matrix: numImages x numMetrics (ensure dimensions match!)
% For illustration, let's assume results is already defined as a matrix.
% If not, you can create it like this:
% results = rand(numImages, numel(metric_names)); % Example data

%fid = fopen('SwinFusion_M3FD_all_metrics.csv', 'w');

% Write header
fprintf(fid, 'S.No.');
for k = 1:numel(metric_names)
    fprintf(fid, ',%s', metric_names{k});
end
fprintf(fid, '\n');

% Write data rows
for i = 1:numImages
    fprintf(fid, '%d', i);
    for k = 1:numel(metric_names)
        fprintf(fid, ',%.6f', results(i,k));
    end
    fprintf(fid, '\n');
end

fclose(fid);

disp('All metrics saved to all_metrics_per_image.txt');
