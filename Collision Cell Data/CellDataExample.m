% This script demonstrates the use of CellDataReader to plot the average
% occupancy of the collision cells.

close all
clear all
clc

format compact

% open a new SRD output file reader
in = CellDataReader('example.srd')
t = getAllFrameTimes(in);

% get occupancy data
occ = getAllOccupancyData(in);

% print out the first few moments of the distribution
fprintf("occupancy mean = %f\n", mean(occ(:)));
fprintf("occupancy std  = %f\n", std(occ(:)));
fprintf("occupancy skew = %f\n", skewness(occ(:)));

% make a histogram
figure
histogram(occ(:), 50, 'Normalization', 'count')
title('Cell Occupancy')

% get velocity data
vel = 1000 * getAllVelocityData(in); % convert nm/ps -> m/s

% concatenate all frames together
vall = reshape(vel, [in.nframes * in.ncells 3]);

figure
histogram(dot(vall',vall'), 200, 'Normalization', 'count')
title('Cell Velocity')
xlim([0, 1000])
ylim([0, 35000])
% Note: this is not the same as average SRD particle velocity!

% get velocity field for first frame
vel1 = 1000 * getVelocityData(in, 1); % convert nm/ps -> m/s

% get center positions of cells
cellsize = getCellSize(in,1);
cellsX = ((1:in.cellCount(1)) - 0.5) * cellsize(1);
cellsY = ((1:in.cellCount(2)) - 0.5) * cellsize(2);
cellsZ = ((1:in.cellCount(3)) - 0.5) * cellsize(3);
[X, Y, Z] = meshgrid(cellsX, cellsY, cellsZ);

% size of simulation box
boxsize = cellsize.*in.cellCount';

% plot a vector field
figure
quiver3(X(:),Y(:),Z(:),vel1(:,1),vel1(:,2),vel1(:,3),'AutoScaleFactor',2)
axis([0, boxsize(1), 0, boxsize(2), 0, boxsize(3)])
title('Frame 1 Velocity Field with Streamlines')

% starting points for streamlines
sx = linspace(0,boxsize(1),5) + boxsize(1) / 8; sx = sx(1:end-1);
sy = linspace(0,boxsize(2),5) + boxsize(2) / 8; sy = sy(1:end-1);
sz = ones(1,4)*boxsize(3) / 2;
[SX, SY, SZ] = meshgrid(sx, sy, sz);

% add streamlines from the midplane
U = griddata(X(:),Y(:),Z(:),vel1(:,1),X,Y,Z );
V = griddata(X(:),Y(:),Z(:),vel1(:,2),X,Y,Z );
W = griddata(X(:),Y(:),Z(:),vel1(:,3),X,Y,Z );

hlines = streamline(X,Y,Z,U,V,W,SX(:),SY(:),SZ(:));
set(hlines,'LineWidth',2,'Color','r')

% make another vector plot with coneplot
figure
sx = linspace(0,boxsize(1),(10+1)) + boxsize(1) / (10*2); sx = sx(1:end-1);
sy = linspace(0,boxsize(2),(10+1)) + boxsize(2) / (10*2); sy = sy(1:end-1);
sz = linspace(0,boxsize(3),(5+1)) + boxsize(3) / (5*2); sz = sz(1:end-1);
[SX, SY, SZ] = meshgrid(sx, sy, sz);
hcone = coneplot(X,Y,Z,U,V,W,SX(:),SY(:),SZ(:), 2);
axis([0, boxsize(1), 0, boxsize(2), 0, boxsize(3)])
hcone.FaceColor = 'blue';
hcone.EdgeColor = 'none';
hcone.DiffuseStrength = 0.8;
camlight 
lighting gouraud
grid on
title('Frame 1 Velocity Field')

% you can also get cells indexed by their cell id
% vcells = reshape(vel1, [in.cellCount 3]); % vcells(ci, cj, ck, dim)

% close the file reader
clear in
