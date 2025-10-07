import torch
import torch.nn as nn
from .attention import Attention
from utils.Expression.defs import ABOVE_BELOW_COMMANDS, ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW

class SAN_decoder(nn.Module):

    def __init__(self, params):
        super(SAN_decoder, self).__init__()

        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channels']
        self.word_num = params['word_num']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        self.struct_num = params['struct_num'] # 7: below, above, inside, right, sub, sup, L-sup
        # the relation in the other they are encoded in the hybrid tree
        self.struct_dict = self.params['words'].encode(['above', 'below', 'sub', 'sup', 'L-sup', 'inside', 'right'])
        self.STRUCT_ID = self.params['words'].encode(['struct'])[0]
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet']['conv1_stride']

        self.threshold = params['hybrid_tree']['threshold']

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)

        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size)

        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)

        # structure gru
        self.struc_input_gru = nn.GRUCell(self.input_size, self.hidden_size)

        # attention
        self.word_attention = Attention(params)

        # state to word/struct
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_embedding_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.word_convert = nn.Linear(self.hidden_size // 2, self.word_num)

        self.struct_convert = nn.Linear(self.hidden_size // 2, self.struct_num)

        """ child to parent """
        self.c2p_input_gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.c2p_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)

        self.c2p_attention = Attention(params)

        self.c2p_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_word_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_relation_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.c2p_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.c2p_convert = nn.Linear(self.hidden_size // 2, self.word_num)

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features, images_mask):

        height, width = cnn_features.shape[2:]
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio].contiguous()

        word_alpha_sum = torch.zeros((1, 1, height, width)).to(device=self.device)
        struct_alpha_sum = torch.zeros((1, 1, height, width)).to(device=self.device)

        if False:
            pass

        else:
            word_embedding = self.embedding(torch.ones(1).long().to(device=self.device))
            struct_list = []
            parent_hidden = self.init_hidden(cnn_features, images_mask)

            iter = 0
            cid, pid = 0, 0
            p_re = 'Start'
            word = torch.LongTensor([1])
            result = [['<s>', 0, -1, 'root']]

            while iter < 400:
                iter += 1

                # word
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, word_hidden_first,
                                                                                   word_alpha_sum, images_mask)
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted

                word_prob = self.word_convert(word_out_state)
                p_word = word
                _, word = word_prob.max(1)
                word_str = self.params['words'].words_index_dict[word.item()]

                if word_str == 'struct':
                    # this follows a symbol or construct (frac, sum, etc.)
                    struct_prob = self.struct_convert(word_out_state)

                    structs = torch.sigmoid(struct_prob)

                    # p_word_str = self.params['words'].words_index_dict[p_word.item()]
                    # if p_word_str == '\\row':
                    #     order = range(structs.shape[1])
                    # else:    
                    #     order = range(structs.shape[1]-1, -1, -1)

                    # for num in order:
                    for num in range(structs.shape[1]-1, -1, -1):
                        if structs[0][num] > self.threshold:
                            struct_list.append((self.struct_dict[num], hidden, p_word, p_id, word_alpha_sum))
                    if len(struct_list) == 0:
                        break
                    word, parent_hidden, p_word, pid, word_alpha_sum = struct_list.pop()
                    word_embedding = self.embedding(torch.LongTensor([word]).to(device=self.device))
                    word_str = self.params['words'].words_index_dict[word]
                    p_word_str = self.params['words'].words_index_dict[p_word.item()]
                    if p_word_str == '\\frac':
                        if word_str == 'above':
                            p_re = 'Above'
                        elif word_str == 'below':
                            p_re = 'Below'
                        else:
                            # illegal relation for fraction, neglect
                            pass
                    elif p_word_str == '\\sqrt':
                        if word_str == 'L-sup':
                            p_re = 'l_sup'
                        elif word_str == 'inside':
                            p_re = 'Inside'
                        elif word_str == 'sup':
                            p_re = 'Sup'
                        else:
                            # illegal relation for sqrt, neglect
                            pass
                    elif p_word_str == '\\stack':
                        if word_str == 'inside':
                            p_re = 'Inside'
                        else:
                            # illegal relation for stack, neglect
                            pass
                    elif p_word_str == '\\row':
                        if word_str == 'inside':
                            p_re = 'Inside'
                        if word_str == 'below':
                            p_re = 'Below'
                        else:
                            # illegal relation for stack, neglect
                            pass
                    elif p_word_str in ABOVE_BELOW_COMMANDS:
                        if word_str in ['below', 'sub']:
                            p_re = 'Below'
                        elif word_str in ['above', 'sup']:
                            p_re = 'Above'
                        else:
                            # illegal relation for sum/prod, neglect
                            pass
                    elif p_word_str in ACCENT_COMMANDS_ABOVE:
                        if word_str == 'sup':
                            p_re = 'Sup'
                        elif word_str == 'sub':
                            p_re = 'Sub'
                        elif word_str == 'below':
                            p_re = 'Below'
                        else:
                            # illegal relation for accent above, neglect
                            pass    
                    elif p_word_str in ACCENT_COMMANDS_BELOW:
                        if word_str == 'sup':
                            p_re = 'Sup'
                        elif word_str == 'sub':
                            p_re = 'Sub'
                        elif word_str == 'above':
                            p_re = 'Above'
                        else:
                            # illegal relation for accent above, neglect
                            pass    
                    else:
                        if word_str == 'sub':
                            p_re = 'Sub'
                        elif word_str == 'sup':
                            p_re = 'Sup'
                elif word_str == '<eos>':
                    if len(struct_list) == 0:
                        break

                    word, parent_hidden, p_word, pid, word_alpha_sum = struct_list.pop()
                    word_embedding = self.embedding(torch.LongTensor([word]).to(device=self.device))
                    word_str = self.params['words'].words_index_dict[word]
                    p_word_str = self.params['words'].words_index_dict[p_word.item()]

                    if word_str == 'inside':
                        p_re = 'Inside'
                    elif word_str == 'sub' or (word_str == 'below' and p_word_str in ABOVE_BELOW_COMMANDS):
                        p_re = 'Sub'
                    elif word_str == 'sup' or (word_str == 'above' and p_word_str in ABOVE_BELOW_COMMANDS):
                        p_re = 'Sup'
                    elif word_str == 'above':
                        p_re = 'Above'
                    elif word_str == 'below':
                        p_re = 'Below'
                    elif word_str == 'L-sup':
                        p_re = 'l_sup'
                    elif word_str == 'inside':
                        p_re = 'Inside'
                    elif word_str == 'right':
                        p_re = 'Right'
                else:
                    # symbol or construct head (e.g. \frac, \sum, etc.)
                    if word.item():
                        cid += 1
                        p_id = cid
                        result.append([self.params['words'].words_index_dict[word.item()], cid, pid, p_re])

                    p_re = 'Right'
                    pid = cid
                    word_embedding = self.embedding(word)
                    parent_hidden = hidden.clone()

        return result


    def init_hidden(self, features, feature_mask):

        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)

        return torch.tanh(average)