import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.san_model.decoder.attention import Attention
from utils.Expression.defs import ABOVE_BELOW_COMMANDS, ACCENT_COMMANDS_ABOVE, ACCENT_COMMANDS_BELOW
from utils.Expression.utils import get_allowed_relations

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
        self.structs_str = ['above', 'below', 'sub', 'sup', 'L-sup', 'inside', 'right']
        self.struct_dict = self.params['words'].encode(self.structs_str)
        self.STRUCT_ID = torch.LongTensor(self.params['words'].encode(['struct']))
        self.RIGHT_ID = torch.LongTensor(self.params['words'].encode(['right']))
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

        # Learnable weight for balancing alpha_sum_parent vs alpha (latest symbol)
        alpha_sum_parent_weight_init = params.get('attention', {}).get('alpha_sum_parent_weight', 1.0)
        self.alpha_sum_parent_weight = nn.Parameter(torch.tensor(alpha_sum_parent_weight_init))

        # Learnable decay factor for parent attention (exponential decay over time)
        alpha_decay_init = params.get('attention', {}).get('alpha_decay', 1.0)
        self.alpha_decay = nn.Parameter(torch.tensor(alpha_decay_init))

        # state to word/struct
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_embedding_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)
        self.word_convert = nn.Linear(self.hidden_size // 2, self.word_num)

        self.struct_convert = nn.Linear(self.hidden_size // 2, self.struct_num)

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio'])

    def forward(self, cnn_features, images_mask, images, word_log_priors=None):

        height, width = cnn_features.shape[2:]
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio].contiguous()

        if False:
            pass

        else:
            word_embedding = self.embedding(torch.ones(1).long().to(device=self.device))
            alpha_sum_active = torch.zeros((1, 1, height, width)).to(device=self.device)
            alpha_sum_completed = torch.zeros((1, 1, height, width)).to(device=self.device)
            alpha_prev_init     = torch.zeros((1, 1, height, width)).to(device=self.device)
            alpha_prev = alpha_prev_init
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

                # Compute attention with dual aggregates (return debug values for visualization)
                word_context_vec, word_alpha, alpha_query, alpha_coverage = self.word_attention(
                    cnn_features, word_hidden_first,
                    alpha_sum_completed, alpha_sum_active * self.alpha_sum_parent_weight + alpha_prev, images_mask, return_debug=True)

                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted

                word_prob = self.word_convert(word_out_state)
                word_log_prob = torch.log_softmax(word_prob, dim=1)  # normalize to log P(word|image)
                if word_log_priors is not None:
                    word_log_prob = word_log_prob + word_log_priors  # add per-request symbol adjustments
                p_word = word

                p_word_str = self.params['words'].words_index_dict[p_word.item()]
                _, word = word_log_prob.max(1)
                word_str = self.params['words'].words_index_dict[word.item()]

                if word_str == 'struct':
                    # this follows a symbol or construct (frac, sum, etc.)
                    struct_prob = self.struct_convert(word_out_state)
                    structs = torch.sigmoid(struct_prob)

                    alpha_struct = word_alpha + alpha_prev
                    alpha_sum_active = alpha_sum_active * self.alpha_decay + alpha_struct
                    alpha_prev = alpha_prev_init

                    # Push active structures to stack (in reverse order)
                    # Each item stores: (relation, hidden_state, parent_word, parent_id)
                    relations = get_allowed_relations(p_word_str)
                    for rel_str in reversed(relations):
                        rel_num = self.structs_str.index(rel_str)
                        if structs[0][rel_num] > self.threshold:
                            struct_word = torch.LongTensor([self.struct_dict[rel_num]])
                            struct_list.append((struct_word, hidden, p_word, pid, alpha_sum_active, alpha_struct))
                    if len(struct_list) == 0:
                        # todo: what to do here?
                        break

                    # Pop first structure from stack
                    word, parent_hidden, p_word, pid, alpha_sum_active, alpha_struct = struct_list.pop()
                    if word == self.RIGHT_ID:
                        # completed the struct (e.g. \frac or a sup)
                        alpha_sum_active = (alpha_sum_active - alpha_struct) / self.alpha_decay
                        alpha_prev = alpha_struct

                    word_embedding = self.embedding(word).to(device=self.device)
                    word_str = self.params['words'].words_index_dict[word.item()]
                    p_word_str = self.params['words'].words_index_dict[p_word.item()]

                    allowed_relations = get_allowed_relations(p_word_str)
                    if word_str in allowed_relations:
                        p_re = word_str
                    else:
                        pass

                elif word_str == '<eos>':
                    if len(struct_list) == 0:
                        break

                    alpha_sum_completed = alpha_sum_completed + alpha_prev
                    alpha_prev = alpha_prev_init

                    # Pop next structure from stack
                    word, parent_hidden, p_word, pid, alpha_sum_active, alpha_struct = struct_list.pop()
                    if word == self.RIGHT_ID:
                        # completed the struct (e.g. \frac or a sup)
                        alpha_sum_active = (alpha_sum_active - alpha_struct) / self.alpha_decay
                        alpha_prev = alpha_struct

                    word_embedding = self.embedding(word).to(device=self.device)
                    word_str = self.params['words'].words_index_dict[word.item()]
                    p_word_str = self.params['words'].words_index_dict[p_word.item()]
                    p_re = word_str # above, below, sub, sup, etc
                else:
                    # symbol or construct head (e.g. \frac, \sum, etc.)
                    if word.item():
                        cid += 1
                        result.append([self.params['words'].words_index_dict[word.item()], cid, pid, p_re])

                    # in default left-to-right, the current symbol is the parent of the next
                    p_re = 'right'
                    pid = cid 
                    word_embedding = self.embedding(word).to(device=self.device)
                    parent_hidden = hidden.clone()
                    alpha_sum_completed = alpha_sum_completed + alpha_prev
                    alpha_prev = word_alpha

                # show attention heatmap
                # from utils.show_attention import visualize_attention
                # image = images[0,0,:,:]
                # alpha = word_alpha[0,:,:]
                # visualize_attention(image, alpha, alpha_query, alpha_coverage, iter, word_str)
                                        
        return result


    def init_hidden(self, features, feature_mask):

        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)

        return torch.tanh(average)
    
