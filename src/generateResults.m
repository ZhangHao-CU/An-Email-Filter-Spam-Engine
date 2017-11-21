predict_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/save_data';
test_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/test_data';
label = load(fullfile(predict_path, 'test_labels.mat'));
label = label.test_label;
fd = fopen(fullfile(test_data_path, 'test.txt'));
test_names = textscan(fd, '%s');
fclose(fd);
test_names = test_names{1,1};
a = string(zeros(800, 1));
for i =1:800
    a(i,1) = test_names{i,1};
end
fd = fopen(fullfile(predict_path, 'results.csv'),'w');
fprintf(fd,'email_id,labels\n');
for i = 1:800
    fprintf(fd,'%d,%d\n',i,label(i,:));
end
fclose(fd);
%csvwrite(fullfile(predict_path, 'result_bow_knn.csv'), l);