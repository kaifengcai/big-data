clc; clear;
%hog
I2 = imread('original/train/dribbling/1.jpg');
[hog1,visualization] = extractHOGFeatures(I2,'CellSize',[32 32]);
subplot(1,2,1);
imshow(I2);
subplot(1,2,2);
plot(visualization);

figure;
imshow(I2); 
hold on;
plot(visualization);

%sift

sift(I2,3,5,1.1)

%lbp
I = rgb2gray(I2);
%Extract unnormalized LBP features so that you can apply a custom normalization.

lbpFeatures = extractLBPFeatures(I,'CellSize',[32 32],'Normalization','None');
%Reshape the LBP features into a number of neighbors -by- number of cells array to access histograms for each individual cell.

numNeighbors = 8;
numBins = numNeighbors*(numNeighbors-1)+3;
lbpCellHists = reshape(lbpFeatures,numBins,[]);
%Normalize each LBP cell histogram using L1 norm.

lbpCellHists = bsxfun(@rdivide,lbpCellHists,sum(lbpCellHists));
%Reshape the LBP features vector back to 1-by- N feature vector.

lbpFeatures = reshape(lbpCellHists,1,[]);
figure
bar([lbpCellHists]','grouped')
title('LBP Histogram')

% color histagram
Red = I2(:,:,1);
Green = I2(:,:,2);
Blue = I2(:,:,3);
 %Get histValues for each channel
[yRed, x] = imhist(Red);
[yGreen, x] = imhist(Green);
[yBlue, x] = imhist(Blue);
    %Plot them together in one plot
figure
plot(x, yRed, 'Red', x, yGreen, 'Green', x, yBlue, 'Blue');
title('color histagram')



