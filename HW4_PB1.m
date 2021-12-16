[I,map]=imread('image.gif');
G=ind2gray(I,map);
% imagesc(I); colormap(map);
% imagesc(G); colormap(gray); 

% For generating SNR and CR lists
blockSizes = [2; 4; 8; 16; 32; 64];
[snrList, crList] = generateStats(G, blockSizes, 0); % the last argument to apply DPCM

% For generating an image separately
% the 2nd last argument to apply DPCM, 
% the last argument to generate an image
[snrValue, cr] = applyDCT(G, 2, 0, 1); 

% For plotting the SNR graph
figure(1);
plot(blockSizes, snrList);
title('Line Plot between N and SNR', 'FontSize',14, 'FontWeight','bold');
xlabel('N', 'FontSize', 14,'FontWeight','bold');
ylabel('Set of SNR values', 'FontSize', 14,'FontWeight','bold');

function [snrList, crList] = generateStats(G, blockSizes, applyDpcm)
    snrList = zeros(1, length(blockSizes));
    crList = zeros(1, length(blockSizes));

    for i = 1:length(blockSizes)
        [snrList(i), crList(i)] = applyDCT(G, blockSizes(i), applyDpcm, 0);
    end
end

function [snrValue, cr] = applyDCT(G, blockSize, applyDpcm, showImage)
    G = double(G);
    dG = blockproc(G, [blockSize blockSize], @(blkStruct) dct2(blkStruct.data));
    if applyDpcm
        [~, dcDequantized] = quantizeDCWithDpcm(dG, blockSize);
    else
        [~, dcDequantized] = quantizeDC(dG, blockSize);
    end
    
    diagonals = 2:blockSize;
    [~, acDequantized] = quantizeAC(dG, blockSize, diagonals);
    dequantizedImage = dequantizeImage(dG, blockSize, dcDequantized, acDequantized, diagonals);
    G_hat = blockproc(dequantizedImage, [blockSize blockSize], @(blkStruct) idct2(blkStruct.data));
    
    snrValue = snr(G, G - G_hat);
    cr = calculateCR(G, blockSize);

    if showImage
        imagesc(G_hat); 
        colormap(gray);
    end
end

function [dcQuantized, dcDequantized] = quantizeDC(matrix, blockSize)
    matrixSize = size(matrix);
    level1BlockSize = matrixSize(1)/blockSize;
    level2BlockSize = matrixSize(2)/blockSize;
    dcTermsSize = level1BlockSize * level2BlockSize; 
    dcTerms = zeros(1, dcTermsSize, 'double');
    
    for i = 1:level1BlockSize
        for j = 1:level2BlockSize
            value = matrix((i - 1) * blockSize + 1, (j - 1) * blockSize + 1);
            key = (i - 1) * level2BlockSize  + j;
            dcTerms(key) = value;
        end
    end

    dFirst = floor(min(dcTerms));
    dLast = ceil(max(dcTerms) + power(10, -6));
    [dLevels, rLevels] = calcUniformIntervals(dFirst, dLast, 8);
    [dcQuantized, dcDequantized] = quanDequantArray(dcTerms, dLevels, rLevels);
end

function [dcQuantized, dcDequantized] = quantizeDCWithDpcm(matrix, blockSize)
    matrixSize = size(matrix);
    level1BlockSize = matrixSize(1)/blockSize;
    level2BlockSize = matrixSize(2)/blockSize;
    dcTermsSize = level1BlockSize * level2BlockSize; 
    dcTerms = zeros(1, dcTermsSize, 'double');
    
    for i = 1:level1BlockSize
        for j = 1:level2BlockSize
            value = matrix((i - 1) * blockSize + 1, (j - 1) * blockSize + 1);
            key = (i - 1) * level2BlockSize  + j;
            if j == 1
                dcTerms(key) = value;
            else
                prevValue = matrix((i - 1) * blockSize + 1, (j - 2) * blockSize + 1);
                dcTerms(key) = value - prevValue;
            end 
        end
    end

    dFirst = floor(min(dcTerms));
    dLast = ceil(max(dcTerms) + power(10, -6));
    [dLevels, rLevels] = calcUniformIntervals(dFirst, dLast, 8);
    [dcQuantized, dcDequantized] = quanDequantArray(dcTerms, dLevels, rLevels);

    for i = 1:level1BlockSize
        for j = 1:level2BlockSize
            key = (i - 1) * level2BlockSize + j;
            if j > 1
                dcDequantized(key) = dcDequantized(key) + dcDequantized(key-1);
            end
        end
    end
end

function [acQuantized, acDequantized] = quantizeAC(matrix, blockSize, diagonals)
    matrixSize = size(matrix);
    level1BlockSize = matrixSize(1)/blockSize;
    level2BlockSize = matrixSize(2)/blockSize;
    totalDiagonalTerms = blockSize * blockSize - 1;
    acTermsSize = level1BlockSize * level2BlockSize * totalDiagonalTerms; 
    acTerms = zeros(1, acTermsSize, 'double');
    keyCounter = 1;
    
    for i = 1:level1BlockSize
        for j = 1:level2BlockSize
            asc = true;
            for d = 1:length(diagonals)
                for t = 1:diagonals(d)
                    if asc
                        rowIndex = t;
                        colIndex = diagonals(d)-t+1;
                    else
                        colIndex = t;
                        rowIndex = diagonals(d)-t+1;
                    end

                    value = matrix((i - 1) * blockSize + rowIndex, (j - 1) * blockSize + colIndex);
                    acTerms(keyCounter) = value;

                    % reverse
                    rowIndexR = blockSize - rowIndex + 1;
                    colIndexR = blockSize - colIndex + 1;

                    valueR = matrix((i - 1) * blockSize + rowIndexR, (j - 1) * blockSize + colIndexR);
                    keyR = ((i - 1) * level2BlockSize + j) * totalDiagonalTerms - mod(keyCounter, totalDiagonalTerms);
                    acTerms(keyR) = valueR;

                    keyCounter = keyCounter + 1;
                end
                asc = ~asc;
            end
            
            keyLast = ((i - 1) * level2BlockSize + j) * totalDiagonalTerms;
            valueLast = matrix((i - 1) * blockSize + blockSize, (j - 1) * blockSize + blockSize);
            acTerms(keyLast) = valueLast;

            keyCounter = keyLast + 1;
        end
    end

    dFirst = floor(min(acTerms));
    dLast = ceil(max(acTerms) + power(10, -6));
    termNumber = floor((blockSize * blockSize - 1)/10);

    level4Terms = zeros(1, level1BlockSize * level2BlockSize * termNumber, 'double');
    level2Terms = zeros(1, level1BlockSize * level2BlockSize * termNumber, 'double');

    level4TermIndex = 1;
    level2TermIndex = 1;
    for n = 1:acTermsSize
        % level4: 1 = 1, 2 = 2, 6 = 6, 7 = 64, 8 = 65, 9 = 66
        % level2: 1 = 7, 2 = 8, 6 = 12, 7 = 70, 8 = 71, 9 = 72 
        if rem(n, blockSize * blockSize - 1) <= termNumber && rem(n, blockSize * blockSize - 1) >= 1
            level4Terms(level4TermIndex) = acTerms(n);
            level4TermIndex = level4TermIndex + 1;
        elseif rem(n, blockSize * blockSize - 1) <= termNumber * 2 && rem(n, blockSize * blockSize - 1) >= 1
            level2Terms(level2TermIndex) = acTerms(n);
            level2TermIndex = level2TermIndex + 1;
        end
    end

    [dLevels4, rLevels4] = calcUniformIntervals(dFirst, dLast, 4);
    [dLevels2, rLevels2] = calcUniformIntervals(dFirst, dLast, 2);

    [acQuantized4, acDequantized4] = quanDequantArray(level4Terms, dLevels4, rLevels4);
    [acQuantized2, acDequantized2] = quanDequantArray(level2Terms, dLevels2, rLevels2);
    
    acQuantized = zeros(1, acTermsSize, 'double'); 
    acDequantized = zeros(1, acTermsSize, 'double');

    level4TermIndex = 1;
    level2TermIndex = 1;
    for n = 1:acTermsSize
        % level4: 1 = 1, 2 = 2, 6 = 6, 7 = 64, 8 = 65, 9 = 66
        % level2: 1 = 7, 2 = 8, 6 = 12, 7 = 70, 8 = 71, 9 = 72 
        if rem(n, blockSize * blockSize - 1) <= termNumber && rem(n, blockSize * blockSize - 1) >= 1
            acQuantized(n) = acQuantized4(level4TermIndex);
            acDequantized(n) = acDequantized4(level4TermIndex);
            level4TermIndex = level4TermIndex + 1;
        elseif rem(n, blockSize * blockSize - 1) <= termNumber * 2 && rem(n, blockSize * blockSize - 1) >= 1
            acQuantized(n) = acQuantized2(level2TermIndex);
            acDequantized(n) = acDequantized2(level2TermIndex);
            level2TermIndex = level2TermIndex + 1;
        end
    end
end

function [dequantizedImage] = dequantizeImage(matrix, blockSize, dcDequantized, acDequantized, diagonals)
    matrixSize = size(matrix);
    level1BlockSize = matrixSize(1)/blockSize;
    level2BlockSize = matrixSize(2)/blockSize;
    dequantizedImage = zeros(matrixSize(1), matrixSize(2));
    totalDiagonalTerms = blockSize * blockSize - 1;
    keyCounter = 1;

    for i = 1:matrixSize(1)/blockSize
        for j = 1:matrixSize(2)/blockSize
            dcKey = (i - 1) * matrixSize(2)/blockSize + j;
            dequantizedImage((i - 1) * blockSize + 1, (j - 1) * blockSize + 1) = dcDequantized(dcKey);
        end
    end
    
    for i = 1:level1BlockSize
        for j = 1:level2BlockSize            
            asc = true;
            for d = 1:length(diagonals)
                for t = 1:diagonals(d)
                    if asc
                        rowIndex = t;
                        colIndex = diagonals(d)-t+1;
                    else
                        colIndex = t;
                        rowIndex = diagonals(d)-t+1;
                    end

                    dequantizedImage((i - 1) * blockSize + rowIndex, (j - 1) * blockSize + colIndex) = acDequantized(keyCounter);

                    % reverse
                    rowIndexR = blockSize - rowIndex + 1;
                    colIndexR = blockSize - colIndex + 1;
                    keyR = ((i - 1) * level2BlockSize + j) * totalDiagonalTerms - mod(keyCounter, totalDiagonalTerms);

                    dequantizedImage((i - 1) * blockSize + rowIndexR, (j - 1) * blockSize + colIndexR) = acDequantized(keyR);
                    keyCounter = keyCounter + 1;
                end
                asc = ~asc;
            end
            
            keyLast = ((i - 1) * level2BlockSize + j) * totalDiagonalTerms;
            dequantizedImage((i - 1) * blockSize + blockSize, (j - 1) * blockSize + blockSize) = acDequantized(keyLast);
                    
            keyCounter = keyLast + 1;
        end
    end
end

function [dLevels, rLevels] = calcUniformIntervals(dFirst, dLast, levelSize)
    dLevels = zeros(1, levelSize + 1, 'double');
    rLevels = zeros(1, levelSize, 'double');
    dLevels(1) = dFirst;
    dLevels(levelSize + 1) = dLast;
    
    delta = (dLast - dFirst)/levelSize;
    for n = 2:levelSize
        dLevels(n) = dLevels(n-1) + delta;
    end
    
    for n = 1:levelSize
        rLevels(n) = (dLevels(n) + dLevels(n+1))/2;
    end
end

function [quantized, dequantized] = quanDequantArray(array, dLevels, rLevels)
    arraySize = length(array);
    quantized = zeros(1, arraySize, 'double');
    dequantized = zeros(1, arraySize, 'double');

    for i = 1:arraySize
        dIndex = findLevelIndex(dLevels, array(i));
        quantized(i) = dIndex;
        dequantized(i) = rLevels(dIndex);
    end    
end

function index = findLevelIndex(levels, value)
    for i = 1:length(levels)-1
        if value == levels(length(levels))
            index = length(levels) - 1;
            break;
        elseif value >= levels(i) && value < levels(i+1)
            index = i;
            break;
        end
    end
end

function cr = calculateCR(matrix, blockSize)
    % imageTotalPixels = width * height
    matrixSize = size(matrix);
    imageTotalPixels = matrixSize(1) * matrixSize(2);
    numberOfTotalBlocks = matrixSize(1) * matrixSize(2) / blockSize / blockSize;
    
    % dcTotalBits = numberOfTotalBlocks * bits/dc
    dcTotalBits =  numberOfTotalBlocks * log2(8);
    
    acTermNumber = floor((blockSize * blockSize - 1)/10);
    acTotalBits = numberOfTotalBlocks * acTermNumber * log2(4) + numberOfTotalBlocks * acTermNumber * log2(2);
    quantizedTotalBits = dcTotalBits + acTotalBits;
%     bitrate = quantizedTotalBits / imageTotalPixels; % bits/px
    cr = imageTotalPixels * 8 / quantizedTotalBits;
end
