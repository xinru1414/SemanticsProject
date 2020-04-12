"""
Feb 2020
Xinru Yan
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class BiLSTM(nn.Module):
    def __init__(self, config, dl, device):
        super(BiLSTM, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(num_embeddings=dl.vocab_size, embedding_dim=self.config.word_emb_dim, padding_idx=dl.PAD_IDX)
        self.word_embeddings.weights = nn.Parameter(nn.init.xavier_uniform_(self.word_embeddings.weight),
                                                    requires_grad=False)

        if self.config.mode == 'None':
            self.bilstm = nn.LSTM(input_size=self.config.word_emb_dim, hidden_size=self.config.hidden_size,
                                   num_layers=self.config.hidden_layers, bidirectional=True, batch_first=True,
                                   dropout=self.config.dropout)
        elif self.config.mode == 'SP' or self.config.mode == 'SM':
            self.bilstm = nn.LSTM(input_size=self.config.word_emb_dim + 1, hidden_size=self.config.hidden_size,
                                  num_layers=self.config.hidden_layers, bidirectional=True, batch_first=True,
                                  dropout=self.config.dropout)
        else:
            self.bilstm = nn.LSTM(input_size=self.config.word_emb_dim + 2, hidden_size=self.config.hidden_size,
                                  num_layers=self.config.hidden_layers, bidirectional=True, batch_first=True,
                                  dropout=self.config.dropout)

        self.fc = nn.Linear(self.config.hidden_size * self.config.hidden_layers * 2, dl.face_size)

        self.dropout = nn.Dropout(self.config.dropout)
        self.to(device)

    def forward(self, X, SM, SP):
        seq_length = X.shape[1]
        # X.shape = [batch_size, seq_length]
        sent_embedded = self.word_embeddings(X)

        # embedded.shape = [batch_size, seq_length, word_emb_dim]
        if self.config.mode == 'SP':
            feature = SP.repeat_interleave(seq_length).view((self.config.batch_size, seq_length))
            embedded = torch.cat((sent_embedded, feature.unsqueeze(2).float()), 2)
        elif self.config.mode == 'SM':
            feature = SP.repeat_interleave(seq_length).view((self.config.batch_size, seq_length))
            embedded = torch.cat((sent_embedded, feature.unsqueeze(2).float()), 2)
        elif self.config.mode == 'ALL':
            feature_1 = SP.repeat_interleave(seq_length).view((self.config.batch_size, seq_length))
            feature_2 = SM.repeat_interleave(seq_length).view((self.config.batch_size, seq_length))
            embedded = torch.cat((sent_embedded, feature_1.unsqueeze(2).float(), feature_2.unsqueeze(2).float()), 2)
        else:
            embedded = sent_embedded
        # print(f'shape after dup {SM.shape}')
        # print(f'duplicating sp shape')
        #
        # print(f'shape after dup {SP.shape}')
        #
        # embedded = torch.cat((sent_embedded, SM.unsqueeze(2), SP.unsqueeze(2)), 2)

        # lstm_out [batch_size, seq_length, 2 * hidden_size]
        # h_n = [num_layers * 2, batch_size, hidden_size]
        # c_n = [num_layers * 2, batch_size, hidden_size]
        lstm_out, (h_n, c_n) = self.bilstm(embedded)

        final_feature_map = self.dropout(h_n)

        # final_feature_map.shape = [batch_size, self.config.hidden_size * self.config.hidden_layers * 2]
        final_feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.fc(final_feature_map)
        return final_out

    def save(self):
        save_best_path = self.config.save_best
        torch.save(self.state_dict(), save_best_path)

    def load(self):
        load_best_path = self.config.save_best
        self.load_state_dict(torch.load(load_best_path))