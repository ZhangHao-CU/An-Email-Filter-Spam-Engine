clear
clc
train_spam_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/train_data/spam';
train_ham_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/train_data/ham';
train_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/train_data';
test_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/test_data';
vectorized_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/vectorized_data';
%**************************************************************************
%load training set of hams and spams emails text
%**************************************************************************
train_spams = load(fullfile(vectorized_data_path,'train_spams.mat'));
train_spams = train_spams.train_spams;
train_hams = load(fullfile(vectorized_data_path,'train_hams.mat'));
train_hams = train_hams.train_hams;
%**************************************************************************
%load test data
%**************************************************************************
tests = load(fullfile(vectorized_data_path,'tests.mat'));
tests = tests.tests;

cleanDocuments = [train_spams;train_hams];
train_label = [ones(size(train_spams,1),1);zeros(size(train_hams,1),1)];
%**************************************************************************
%text stemming for train data
%**************************************************************************
%cleanTextData = erasePunctuation(raw_text);
%cleanTextData = lower(cleanTextData);
%cleanDocuments = tokenizedDocument(cleanTextData);
%cleanDocuments = removeWords(cleanDocuments,stopWords);
%cleanDocuments = removeShortWords(cleanDocuments,2);
%cleanDocuments = removeLongWords(cleanDocuments,15);
%cleanDocuments = normalizeWords(cleanDocuments);
%**************************************************************************
%build BOW model/LDA model
%**************************************************************************
bag = bagOfWords(cleanDocuments);
%bag = removeInfrequentWords(bag,2);
%[bag,~] = removeEmptyDocuments(bag);
train_vecs = tfidf(bag,'TFWeight','log','IDFWeight','smooth');
%train_vecs = encode(bag,cleanDocuments);
%train_vecs = [train_vecs,train_label];
%**************************************************************************
%text stemming for test data
%**************************************************************************
%cleanTextData = erasePunctuation(test);
%cleanDocuments = tokenizedDocument(cleanTextData);
%cleanDocuments = removeWords(cleanDocuments,stopWords);
%cleanDocuments = removeShortWords(cleanDocuments,2);
%cleanDocuments = removeLongWords(cleanDocuments,15);
%cleanDocuments = normalizeWords(cleanDocuments);
%**************************************************************************
%pca
%**************************************************************************

%**************************************************************************
%tfidf test vecs
%**************************************************************************
test_vecs = tfidf(bag,tests,'TFWeight','log','IDFWeight','smooth');
%test_vecs = encode(bag,tests);
%**************************************************************************
%pca zscore
%**************************************************************************
vecs = [train_vecs;test_vecs];
vecs = full(vecs);
vecs = zscore(vecs);
%vecs = zscore(vecs);
[~,pca_vecs,hh] = pca(vecs);
%**************************************************************************
%fit model with training data
%**************************************************************************
%KNNmd = fitcknn(train_vecs,train_label,'NumNeighbors',5,'Standardize',1);
%**************************************************************************
%predict the hams or spams
%**************************************************************************
%label = predict(KNNmd,test_vecs);