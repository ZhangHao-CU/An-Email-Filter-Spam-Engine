clear
clc
train_spam_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/train_data/spam';
train_ham_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/train_data/ham';
train_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/train_data';
test_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/test_data';
%**************************************************************************
%load training set of hams and spams emails text
%**************************************************************************
fd = fopen(fullfile(train_data_path, 'spams.txt'));
spam_names = textscan(fd, '%s');
fclose(fd);
fd = fopen(fullfile(train_data_path, 'hams.txt'));
ham_names = textscan(fd, '%s');
fclose(fd);
spam_names = spam_names{1,1};
ham_names = ham_names{1,1};
train_label = [ones(size(spam_names,1),1);zeros(size(ham_names,1),1)];
%train_label = [ones(10,1);zeros(10,1)];
%**************************************************************************
%text stemming for train data
%**************************************************************************
cleanDocuments = load('/Users/zhanghao/Documents/MATLAB/ML/HW3/email/clean_train_data/clean_train_data.mat');
cleanDocuments = cleanDocuments.cleanDocuments;
%**************************************************************************
%build BOW model/LDA model
%**************************************************************************
bag = bagOfWords(cleanDocuments);
train_vecs = tfidf(bag);
%train_vecs = [train_vecs,train_label];
%**************************************************************************
%fit model with training data
%**************************************************************************
KNNmd = fitcknn(train_vecs,train_label,'NumNeighbors',5,'Standardize',1);
%**************************************************************************
%load test data
%**************************************************************************
fd = fopen(fullfile(test_data_path, 'test.txt'));
test_names = textscan(fd, '%s');
fclose(fd);
test_names = test_names{1,1};
raw_test = string(zeros(size(test_names,1),1));%cell(size(test_names,1),1);
for i = 1:size(test_names,1)
    raw_test(i,1) = extractFileText(fullfile(test_data_path,test_names(i,1)));
end
%**************************************************************************
%text stemming for test data
%**************************************************************************
cleanTextData = erasePunctuation(raw_test);
cleanTextData = lower(cleanTextData);
cleanDocuments = tokenizedDocument(cleanTextData);
cleanDocuments = removeWords(cleanDocuments,stopWords);
cleanDocuments = removeShortWords(cleanDocuments,2);
cleanDocuments = removeLongWords(cleanDocuments,15);
cleanDocuments = normalizeWords(cleanDocuments);
%**************************************************************************
%tfidf test vecs
%**************************************************************************
test_vec = tfidf(bag,cleanDocuments);
%**************************************************************************
%predict the hams or spams
%**************************************************************************
label = predict(KNNmd,test_vec);