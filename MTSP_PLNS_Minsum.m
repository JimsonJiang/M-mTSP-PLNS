
function [statisticsOfOPT, statisticsOfTimeConsume, adjustedHis] = MTSP_PLNS_Minsum(dataSet, numOfSalesmen, populationSize, runs)
    % Process Inputs and Initialize Defaults
    data=dataSet();
    if size(data,2)==3
        x=(data(:,2))'; % X coordinate
        y=(data(:,3))'; % Y coordinate
        xy=[x;y]';
        h=pdist(xy);
        dmat=squareform(h);
        dmat=round(dmat);
        dmat=dmat - diag(diag(dmat)); % Distance matrix
        N=length(x); % Number of cities
    else
        dmat=data;
        N=size(dmat,1);
    end

    nSalesmen = numOfSalesmen;
    nBreaks = nSalesmen - 1;
    minTour = 2;
    maxTour = N - 2*(nSalesmen - 1);
    %minTour = floor(N / (nSalesmen + 1)) + 1;
    %maxTour = ceil(N / (nSalesmen - 1)) - 1;
    if (N == 51 && nSalesmen == 10)
        minTour = 5;
        maxTour = 6;
    end
    popSize = populationSize;
    numIter = 1000;
    showProg = 0;
    showResult = 0;
    n = N;
    dims = 2;
    %cumulativeWeights = weightedRandom(n - nBreaks * 2);


    Run = runs;
    OPT = zeros(1,Run);
    OPT_ROUTE = zeros(Run,N);
    OPT_BRK = zeros(Run,nSalesmen-1);
    timeConsume = zeros(1,Run);
    distHistory = zeros(Run,numIter);
    for R=1:Run


    tic;

    % Initialize the Populations
    popRoute = zeros(popSize,n);         % population of routes
    popBreak = zeros(popSize,nBreaks);   % population of breaks
    valueRecord = [0, 0];

    disp('Algorithm starts running.');
    [popRoute, popBreak] = initialPopulation_final(popRoute, popBreak, data, minTour, maxTour, dmat);
    Time = toc; 

    % Select the Colors for the Plotted Routes
    pclr = ~get(0,'DefaultAxesColor');
    clr = [1 0 0; 0 0 1; 0.67 0 1; 0 1 0; 1 0.5 0];
    if nSalesmen > 5
        clr = hsv(nSalesmen);
    end

    % Run the GA
    globalMin = Inf;
    totalDist = zeros(1,popSize);
    % distHistory = zeros(1,numIter);
    tmpPopRoute = zeros(3,n);
    tmpPopBreak = zeros(3,nBreaks);
    newPopRoute = popRoute;
    newPopBreak = popBreak;
    newTotalDist = totalDist;

    if showProg
        pfig = figure('Name','MTSP_GA | Current Best Solution','Numbertitle','off');
    end

    % Evaluate Members of the Population
    for p = 1:popSize
        totalDist(p) = calculateTotalDistance(popRoute(p,:), popBreak(p,:), dmat);
        newTotalDist(p) = totalDist(p);
    end

    % Find the Best Route in the Population
    [minDist,index] = min(totalDist);
    distHistory(R,1) = minDist;
    if minDist < globalMin
        globalMin = minDist;
        optRoute = popRoute(index,:);
        optBreak = popBreak(index,:);
        rng = [[1 optBreak+1];[optBreak n]]';
        if showProg
            % Plot the Best Route
            figure(pfig);
            for s = 1:nSalesmen
                rte = optRoute([rng(s,1):rng(s,2) rng(s,1)]);
                if dims > 2, plot3(xy(rte,1),xy(rte,2),xy(rte,3),'.-','Color',clr(s,:));
                else plot(xy(rte,1),xy(rte,2),'.-','Color',clr(s,:)); end
                title(sprintf('Total Distance = %1.4f, Iteration = %d',minDist,1));
                hold on
            end
            hold off
        end
    end

    MAXtime = n / 3;
    Time = 0;
    iter = 2;
    noImproveIter = 0;
    globalMinIterIdx = iter;
    %tic;
    %% Main loop
    while Time <= MAXtime

        % Algorithm Operators
        isBreak = false;
        for p=1:popSize
            for k = 1:3 % Generate New Solutions

                [tmpPopRoute(k,:), tmpPopBreak(k,:)] = LNSX_Muti(popRoute(p,:), popBreak(p,:), dmat, minTour, maxTour);

                Time = toc;
                if Time > MAXtime
                    newPopRoute((popSize + 3*p - 2):(popSize + 3*p - (3-k)),:) = tmpPopRoute(1:k,:);
                    newPopBreak((popSize + 3*p - 2):(popSize + 3*p - (3-k)),:) = tmpPopBreak(1:k,:);

                    % Evaluate Members of the new Population
                    for p = (popSize + 1):size(newPopRoute, 1)
                        newTotalDist(p) = calculateTotalDistance(newPopRoute(p,:), newPopBreak(p,:), dmat);
                    end

                    [minDist,index] = min(newTotalDist);
                    distHistory(R,iter) = minDist;
                    if minDist < globalMin
                        globalMin = minDist;
                        optRoute = newPopRoute(index,:);
                        optBreak = newPopBreak(index,:);
                        globalMinIterIdx = iter;
                    end

                    isBreak = true;
                    break;
                end

            end

            if (isBreak)
                break;
            end

            newPopRoute((popSize + 3*p - 2):(popSize + 3*p),:) = tmpPopRoute;
            newPopBreak((popSize + 3*p - 2):(popSize + 3*p),:) = tmpPopBreak;   
        end

        if (isBreak)
            break;
        end

        % Evaluate Members of the new Population
        for p = (popSize + 1):size(newPopRoute, 1)
            newTotalDist(p) = calculateTotalDistance(newPopRoute(p,:), newPopBreak(p,:), dmat);
        end

        % Tournament Selection
        random_sequence = randperm(4 * popSize);
        leftIndex = zeros(1, popSize);
        for p = 4:4:4 * popSize
            temRan = random_sequence(p-3:p);
            [minDistance, index] = min(newTotalDist(temRan));
            totalDist(p/4) = minDistance;
            popRoute(p/4,:) = newPopRoute(temRan(index),:);
            popBreak(p/4,:) = newPopBreak(temRan(index),:);
            leftIndex(p/4) = temRan(index);
        end

        [totalDist, uniqueIndices1] = unique(totalDist, 'stable');
        if length(totalDist) < popSize
            newTotalDist(leftIndex) = [];
            newPopRoute(leftIndex,:) = [];
            newPopBreak(leftIndex,:) = [];
            [newTotalDist, uniqueIndices2] = unique(newTotalDist, 'stable');
            popRoute = popRoute(uniqueIndices1, :);
            popBreak = popBreak(uniqueIndices1, :);
            newPopRoute = newPopRoute(uniqueIndices2, :);
            newPopBreak = newPopBreak(uniqueIndices2, :);
            random_sequence = randperm(length(newTotalDist));
            for p=4:4:length(newTotalDist)
                temRan = random_sequence(p-3:p);
                [minDistance, index] = min(newTotalDist(temRan));
                totalDist(end+1) = minDistance;
                popRoute(end+1,:) = newPopRoute(temRan(index),:);
                popBreak(end+1,:) = newPopBreak(temRan(index),:);
                if length(totalDist) == popSize
                    break;
                end
           end
        end
       
        while length(totalDist) < popSize
            random_sequence = randperm(length(totalDist));
            lengthOfTL = length(totalDist);
            for i=1:lengthOfTL
                totalDist(end+1) = totalDist(random_sequence(i));
                popRoute(end+1,:) = popRoute(random_sequence(i),:);
                popBreak(end+1,:) = popBreak(random_sequence(i),:); 
                if length(totalDist) == popSize
                    break;
                end
            end
        end

        newPopRoute = popRoute;
        newPopBreak = popBreak;
        newTotalDist = totalDist;

        % Local search
        if mod(iter, 100) == 0
            for i = 1:popSize
                [~, popRoute(i, :), popBreak(i, :), valueRecord] = ...
                    HRVNS(popRoute(i,:), popBreak(i,:), minTour, maxTour, dmat, valueRecord, Time);
            end

            % Evaluate Members of the Population
            for p = 1:popSize
                totalDist(p) = calculateTotalDistance(popRoute(p,:), popBreak(p,:), dmat);
                newTotalDist(p) = totalDist(p);
            end
            newPopRoute = popRoute;
            newPopBreak = popBreak;
        end

        % Find the Best Route in the Population
        [minDist,index] = min(totalDist);
        distHistory(R,iter) = minDist;
        %distHistory(iter) = totalDist(1);
        %minDist = totalDist(1);
        if minDist < globalMin
            globalMin = minDist;
            optRoute = popRoute(index,:); %
            optBreak = popBreak(index,:); %
            noImproveIter = 0;
            globalMinIterIdx = iter;
            rng = [[1 optBreak+1];[optBreak n]]';
            if showProg
                % Plot the Best Route
                figure(pfig);
                for s = 1:nSalesmen
                    rte = optRoute([rng(s,1):rng(s,2) rng(s,1)]);
                    if dims > 2, plot3(xy(rte,1),xy(rte,2),xy(rte,3),'.-','Color',clr(s,:));
                    else plot(xy(rte,1),xy(rte,2),'.-','Color',clr(s,:)); end
                    title(sprintf('Total Distance = %1.4f, Iteration = %d',globalMin,iter));
                    hold on
                end
                hold off
            end
        else
            noImproveIter = noImproveIter + 1;
        end

        % Population management
        if  mod(noImproveIter, 100) == 0 && noImproveIter ~= 0
            discardNum = ceil(0.9 * popSize);
            initPopRoute = zeros(discardNum, n);         % population of routes
            initPopBreak = zeros(discardNum, nBreaks);   % population of breaks
            [initPopRoute, initPopBreak] = initialPopulation_final(initPopRoute, initPopBreak, data, minTour, maxTour, dmat);

            [uniqueTotalDist, uniqueIndices] = unique(totalDist, 'stable');
            [totalDist, sortedIndices] = sort(uniqueTotalDist);
            sortedUniqueIndices = uniqueIndices(sortedIndices);
            popRoute = popRoute(sortedUniqueIndices, :);
            popBreak = popBreak(sortedUniqueIndices, :);

            popRoute((popSize - discardNum + 1):popSize, :) = initPopRoute;
            popBreak((popSize - discardNum + 1):popSize, :) = initPopBreak;

            % Evaluate Members of the new Population
            for p = (popSize - discardNum + 1):popSize  
                totalDist(p) = calculateTotalDistance(popRoute(p,:), popBreak(p,:), dmat);
            end
        end

        iter = iter + 1;
        Time = toc; % Total time
    end
    
    disp('Algorithm running completed');
    fprintf('%s%d\n', 'The number of iterations to find optSol is: ', globalMinIterIdx);
    fprintf('%s%d\n', 'The total number of iterations is: ', iter);

    if showResult
    % Plots
        figure('Name','MTSP_GA | Results','Numbertitle','off');
        subplot(2,2,1);
        if dims > 2, plot3(xy(:,1),xy(:,2),xy(:,3),'.','Color',pclr);
        else plot(xy(:,1),xy(:,2),'.','Color',pclr); end
        title('City Locations');
        subplot(2,2,2);
        imagesc(dmat(optRoute,optRoute));
        title('Distance Matrix');
        subplot(2,2,3);
        rng = [[1 optBreak+1];[optBreak n]]';
        for s = 1:nSalesmen
            rte = optRoute([rng(s,1):rng(s,2) rng(s,1)]);
            if dims > 2, plot3(xy(rte,1),xy(rte,2),xy(rte,3),'.-','Color',clr(s,:));
            else plot(xy(rte,1),xy(rte,2),'.-','Color',clr(s,:)); end
            title(sprintf('Total Distance = %1.4f',globalMin));
            hold on;
        end
        subplot(2,2,4);
        plot(distHistory(R,:),'b','LineWidth',2);
        title('Best Solution History');
        set(gca,'XLim',[0 numIter+1],'YLim',[0 1.1*max([1 distHistory(R,:)])]);
    end

    OPT(1,R)=globalMin;
    OPT_ROUTE(R,:)=optRoute;
    OPT_BRK(R,:)=optBreak;
    timeConsume(1,R) = Time;
    end
    
    meanOPT = mean(OPT);
    meanTimeConsume = mean(timeConsume);
    maxOPT = max(OPT);
    maxTimeConsume = max(timeConsume);
    minOPT = min(OPT);
    minTimeConsume = min(timeConsume);
    statisticsOfOPT = [OPT, minOPT, maxOPT, meanOPT];
    statisticsOfTimeConsume = [timeConsume, minTimeConsume, maxTimeConsume, meanTimeConsume];
    
    %adjustedHis = distHistory;
    adjustedHis = (distHistory - OPT') ./ (OPT' / 100);
    meanAdjustedHis = mean(adjustedHis, 1);
    adjustedHis = [adjustedHis; meanAdjustedHis];

end
