clc;
clear;
vectorized_data_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/email/vectorized_data';
predict_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/save_data';
best_predict_path = '/Users/zhanghao/Documents/MATLAB/ML/HW3/save_data/best_result';
bag = load(fullfile(vectorized_data_path,'bag.mat'));
bag = bag.bag;
vecs = load(fullfile(vectorized_data_path, 'vecs.mat'));
vecs = vecs.vecs;

train_hams_num = 3107;
train_spams_num = 1265;
test_num = 800;
train_label = [ones(train_spams_num,1);zeros(train_hams_num,1)];
trains_vecs = vecs(1:train_hams_num+train_spams_num, :);
trains_hams_vecs = trains_vecs(1:train_hams_num,:);
trains_spams_vecs = trains_vecs(train_hams_num+1:end,:);
tests_vecs = vecs(train_hams_num+train_spams_num+1:end, :);
%calculate each dimension of features
trains_hams_vecs = sum(trains_hams_vecs);
trains_hams_vecs = trains_hams_vecs + 1;
trains_hams_vecs = trains_hams_vecs/sum(trains_hams_vecs,2);
trains_hams_vecs = log(trains_hams_vecs);

trains_spams_vecs = sum(trains_spams_vecs);
trains_spams_vecs = trains_spams_vecs + 1;
trains_spams_vecs = trains_spams_vecs/sum(trains_spams_vecs,2);
trains_spams_vecs = log(trains_spams_vecs);

prior = [train_hams_num/(train_hams_num+train_spams_num) train_spams_num/(train_hams_num+train_spams_num)];
prior = log(prior);
test_label = zeros(test_num, 1);
for i = 1:test_num
    each_p = zeros(1,2);
    each_p(1,1) = prior(1,1)+tests_vecs(i,:)*trains_hams_vecs';
    each_p(1,2) = prior(1,2)+tests_vecs(i,:)*trains_spams_vecs'; 
    [~,I] = max(each_p);
    test_label(i,1) = I-1;
end
b = find(test_label == 1);
save(fullfile(predict_path,'test_labels.mat'), 'test_label');
best_result = load(fullfile(best_predict_path,'test_labels.mat'));
best_result = best_result.test_label;
c = find(best_result ~= test_label);