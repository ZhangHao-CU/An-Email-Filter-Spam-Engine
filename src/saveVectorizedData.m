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
fd = fopen(fullfile(train_data_path, 'spams.txt'));
spam_names = textscan(fd, '%s');
fclose(fd);
fd = fopen(fullfile(train_data_path, 'hams.txt'));
ham_names = textscan(fd, '%s');
fclose(fd);
fd = fopen(fullfile(test_data_path, 'test.txt'));
test_names = textscan(fd, '%s');
fclose(fd);
spam_names = spam_names{1,1};
ham_names = ham_names{1,1};
test_names = test_names{1,1};
spams = string(zeros(size(spam_names,1),1));
hams = string(zeros(size(ham_names,1),1));
tests = string(zeros(size(test_names,1),1));
for i = 1:size(spam_names,1)
    spams(i,1) = extractFileText(fullfile(train_spam_path,spam_names(i,1)));
end
for i = 1:size(ham_names,1)
    hams(i,1) = extractFileText(fullfile(train_ham_path,ham_names(i,1)));
end
for i = 1:size(test_names,1)
    tests(i,1) = extractFileText(fullfile(test_data_path,test_names(i,1)));
end

cleanTextData = erasePunctuation(spams);
cleanTextData = lower(cleanTextData);
cleanDocuments = tokenizedDocument(cleanTextData);
cleanDocuments = removeWords(cleanDocuments,stopWords);
%cleanDocuments = removeShortWords(cleanDocuments,2);
%cleanDocuments = removeLongWords(cleanDocuments,15);
cleanDocuments = normalizeWords(cleanDocuments);
spams = cleanDocuments;
save(fullfile(vectorized_data_path, 'spams.mat'),'spams');

cleanTextData = erasePunctuation(hams);
cleanTextData = lower(cleanTextData);
cleanDocuments = tokenizedDocument(cleanTextData);
cleanDocuments = removeWords(cleanDocuments,stopWords);
%cleanDocuments = removeShortWords(cleanDocuments,2);
%cleanDocuments = removeLongWords(cleanDocuments,15);
cleanDocuments = normalizeWords(cleanDocuments);
hams = cleanDocuments;
save(fullfile(vectorized_data_path, 'hams.mat'),'hams');

cleanTextData = erasePunctuation(tests);
cleanTextData = lower(cleanTextData);
cleanDocuments = tokenizedDocument(cleanTextData);
cleanDocuments = removeWords(cleanDocuments,stopWords);
%cleanDocuments = removeShortWords(cleanDocuments,2);
%cleanDocuments = removeLongWords(cleanDocuments,15);
cleanDocuments = normalizeWords(cleanDocuments);
tests = cleanDocuments;
save(fullfile(vectorized_data_path, 'tests.mat'),'tests');