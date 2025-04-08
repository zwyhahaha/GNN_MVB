clear; clc; close all;

% Put all the .mat files into a directory named ${instance}
instance = 'train_wp2383';
mpsfname = [instance '.mps'];
info = dir(instance);

% Ensure that fullfile('..', 'mps', mpsfname) points to the mps file
model = gurobi_read(fullfile('..', 'mps', mpsfname));
bidx = find(model.vtype == 'B');

fnames = {info.name}';
Xall = [];
yall = [];

for i = 1:length(fnames)
    fname = fnames{i};
    try
        load(fullfile(instance, fname));
        Xall = [Xall; X]; %#ok
        yall = [yall; I(:, bidx)]; %#ok
    catch
        
    end % End try
end % End for

X = Xall;
y = yall;

data.X = X;
data.y = y;

save([instance, '.mat'], 'data');