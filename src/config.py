'''
March 2020
Xinru Yan
'''
train_file_path = '../data/preprocessed/102_train_prepared_data.pickle'
dev_file_path = '../data/preprocessed/102_test_prepared_data.pickle'
test_file_path = '../data/preprocessed/102_test_prepared_data.pickle'
model_path = '../model/traditional/'
save_best = '../model/bilstm/model_sm.pt'
batch_size = 10
mode = 'SM'
word_emb_dim = 300
hidden_size = 64
hidden_layers = 2
dropout = 0.3
lr = 0.001
max_epochs = 20
random_seed = 9999