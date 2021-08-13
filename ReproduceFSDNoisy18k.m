rng('default')
dataFolder = 'data/Separate_STFT_addNoise_Class';

imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders',true, ...
    'LabelSource', 'foldernames');

trainingNumFiles = 0.7;
[imdsTrain, imdsValidation] = splitEachLabel(imds, trainingNumFiles, 'randomize');

%%
layers = [
    imageInputLayer([224 224 3],"Name","imageinput")
    batchNormalizationLayer("Name","batchnorm1")
    reluLayer("Name","relu1")
    convolution2dLayer([5 5],24,"Name","conv1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu11")
    maxPooling2dLayer([4 2],"Name","maxpool1","Stride",[4 2])
    batchNormalizationLayer("Name","batchnorm2")
    reluLayer("Name","relu2")
    convolution2dLayer([5 5],48,"Name","conv2","Padding","same")
    batchNormalizationLayer("Name","batchnorm22")
    reluLayer("Name","relu22")
    maxPooling2dLayer([4 2],"Name","maxpool2","Stride",[4 2])
    batchNormalizationLayer("Name","batchnorm3")
    reluLayer("Name","relu3")
    convolution2dLayer([5 5],48,"Name","conv3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu33")
    dropoutLayer(0.5,"Name","dropout_1")
    fullyConnectedLayer(64,"Name","fc1")
    dropoutLayer(0.5,"Name","dropout_2")
    fullyConnectedLayer(21,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
%%
net = layers;

%% 
%net.layers_2(1);
inputSize = [224 224 1];

%% Train Network
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%% To automatically resize the validation images without performing further data augmentation, use an augmented image datastore without specifying any additional preprocessing operations.
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% 
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%
nettrain = trainNetwork(augimdsTrain,net,options);