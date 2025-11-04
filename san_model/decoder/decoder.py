import torch
import torch.nn as nn
from .attention import Attention


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
        self.STRUCT_ID = self.params['words'].encode(['struct'])[0]
        self.RIGHT_ID = self.params['words'].encode(['right'])[0]
        self.EOS_ID = self.params['words'].encode(['<eos>'])[0]
        self.SUB_ID = self.params['words'].encode(['sub'])[0]
        self.SUP_ID = self.params['words'].encode(['sup'])[0]
        self.struct_num = params['struct_num'] # 7: below, above, inside, right, sub, sup, L-sup
        self.struct_dict = torch.tensor(
            self.params['words'].encode(['above', 'below', 'sub', 'sup', 'L-sup', 'inside', 'right']),
            device=self.device
        )
        # self.struct_dict = self.params['words'].encode(['above', 'below', 'sub', 'sup', 'L-sup', 'inside', 'right'])
        self.struct_dict_no_right = torch.tensor([s for s in self.struct_dict if s != self.RIGHT_ID],device=self.device)
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet']['conv1_stride']

        self.threshold = params['hybrid_tree']['threshold']

        # init hidden state
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)

        # word embedding
        self.embedding = nn.Embedding(self.word_num, self.input_size)

        # word gru
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_out_gru = nn.GRUCell(self.out_channel, self.hidden_size)

        # attention
        self.word_attention = Attention(params)

        # state to word/struct
        # Linear mappings W_p, W_g, W_t in the article, used before aggregation
        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_embedding_weight = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size // 2)

        # Linear mappings to output space (word/struct)
        self.word_convert = nn.Linear(self.hidden_size // 2, self.word_num)
        self.struct_convert = nn.Linear(self.hidden_size // 2, self.struct_num)

        """ child to parent """
        if self.params['decoder']['inverse']:
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

    def forward(self, cnn_features, labels, images_mask, labels_mask, is_train=True):

        batch_size, num_steps, _ = labels.shape
        height, width = cnn_features.shape[2:]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        struct_probs = torch.zeros((batch_size, num_steps, self.struct_num)).to(device=self.device)
        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio].contiguous()

        if self.params['decoder']['inverse']:
            c2p_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
            c2p_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
            c2p_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)
            c2p_alpha_sum_completed = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        else:
            c2p_probs, c2p_alphas = None, None

        if is_train:
            # c^{alpha}_0 in the article, for all samples in the batch and all lines in the hybrid tree.
            # We need to keep history of all parent hidden states to fetch the correct one when a new parent is activated
            parent_hiddens = torch.zeros((batch_size * (num_steps + 1), self.hidden_size)).to(device=self.device)
            # Initialize with E(X), features from the encoder. c^{alpha}_0 = W E(X), with W a linear transformation to map dimensions (684 to 256)
            parent_hiddens[:batch_size, :] = self.init_hidden(cnn_features, images_mask)
            # c^{alpha}_p in the article, but for the reversed model (child-to-parent). So the parent state corresponding to the previous relation
            c2p_hidden = torch.zeros((batch_size, self.hidden_size)).to(device=self.device)
            # Syntax-aware attention vector att_{\alpha}(X) per line, accumulated over the parents
            alpha_sum_parents = torch.zeros((batch_size * (num_steps + 1), 1, height, width)).to(device=self.device)
            # the attention weights used per symbol (alpha) or struct (alpha of 'struct' and construct '\frac')
            alpha_currents = torch.zeros((batch_size * (num_steps + 1), 1, height, width)).to(device=self.device)
            # attention vector to penalise covered parts of the image.
            alpha_sum_completed = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
            # attention vector of previous symbol
            alpha_current = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
            
            # Iterate over the lines in the hybrid tree
            for i in range(num_steps):
                
                parent_ids = labels[:,i,2].clone()
                current_type = labels[:,i,1].clone()
                current_parent_type = labels[:,i,3].clone()

                for item in range(len(parent_ids)):
                    parent_ids[item] = parent_ids[item] * batch_size + item
                # retrieve hidden states c^{\alpha}_0 for the current line from the history (parent_hiddens)    
                parent_hidden = parent_hiddens[parent_ids,:].contiguous()
                # syntax-aware attention vector, Att_{\alpha}(X)
                alpha_sum_parent = alpha_sum_parents[parent_ids, :, :, :].contiguous()
                alpha_current = alpha_currents[parent_ids, :, :, :].contiguous()

                # Partner state, c^{\alpha}_p in the article. 
                # Set to the latest generated terminal (teacher forcing strategy)
                word_embedding = self.embedding(labels[:, i, 3])

                # GRU-alpha
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)
                
                # Attention mechanism. word_context_vec is \Omega in the article
                # If parent is a relation, then do not use the alpha_current as it is already included in alpha_sum_parent
                alpha_relation_mask = torch.isin(current_parent_type, self.struct_dict).float().view(batch_size, 1, 1, 1)
                alpha = alpha_current * (1 - alpha_relation_mask)                
                word_context_vec, word_alpha = self.word_attention(
                    cnn_features, word_hidden_first,
                    alpha_sum_completed, alpha_sum_parent + alpha, images_mask)
                
                # GRU-beta
                # hidden is c^{\alpha}_{\beta} in the article
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                # map c^{\alpha}_{\beta}, c^{\alpha}_p, \Omega before aggregation
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                # update word_alpha_sum for structs            
                alpha_struct_mask = (current_type == self.STRUCT_ID).float().view(batch_size, 1, 1, 1)
                alpha_relation_not_right_mask = torch.isin(current_parent_type, self.struct_dict_no_right).float().view(batch_size, 1, 1, 1)
                alpha_right_relation_mask = (current_parent_type == self.RIGHT_ID).float().view(batch_size, 1, 1, 1)
                alpha_not_eos_mask = (current_type != self.EOS_ID).float().view(batch_size, 1, 1, 1)

                # Update sum of active parents as follows:
                # - If current match is 'struct':
                #      e.g. previous = '\frac', current = 'struct'.
                #      then add weights of this 'struct' and the previous '\fact'
                # - ElseIf parent is 'right'
                #      we are moving out of a construct, so
                #      subtract current alpha ('struct' + '\frac')
                alpha_sum_parent = alpha_sum_parent\
                        + alpha_struct_mask * (word_alpha + alpha_current)\
                        - (1-alpha_struct_mask) * alpha_right_relation_mask * alpha_current

                # Update weight of completed symbols as follows
                # - If current match is 'struct' or parent is a relation other than 'right'
                #   no update
                # - else (parent is 'right' or a symbol)
                #   the struct or symbol is completed, so add its current alpha
                alpha_sum_completed = alpha_sum_completed + (1-alpha_struct_mask) * (1-alpha_relation_not_right_mask) * alpha_current

                # Update current alpha as follows:
                # - If current match is 'struct':
                #   Combine word_alpha with word_alpha of the previous symbol (e.g. '\frac')
                # - If current match is 'eos';
                #   Set to zero. (Not really needed, as an 'eos' is never a parent)
                # - Else (current match is a symbol)
                #   Set to current attention vector (word_alpha)
                alpha_current = alpha_not_eos_mask * (word_alpha + alpha_struct_mask * alpha_current)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted

                if i != num_steps - 1:
                    # update history
                    parent_hiddens[(i+1)*batch_size:(i+2)*batch_size,:] = hidden
                    alpha_sum_parents[(i+1)*batch_size:(i+2)*batch_size, :, :, :] = alpha_sum_parent
                    alpha_currents[(i+1)*batch_size:(i+2)*batch_size, :, :, :] = alpha_current

                word_prob = self.word_convert(word_out_state)
                struct_prob = self.struct_convert(word_out_state)
                word_probs[:, i] = word_prob
                struct_probs[:, i] = struct_prob
                word_alphas[:, i] = word_alpha[:,0,:,:]

                if self.params['decoder']['inverse']:
                    c2p_out_state, c2p_alpha, c2p_hidden, c2p_alpha_sum, c2p_alpha_sum_completed = self.child_to_parent(i, labels, cnn_features, c2p_hidden, c2p_alpha_sum_completed, c2p_alpha_sum, images_mask)

                    c2p_prob = self.c2p_convert(c2p_out_state)
                    c2p_probs[:, -(i + 1)] = c2p_prob
                    c2p_alphas[:, -(i + 1)] = c2p_alpha[:,0,:,:]

        else:
            word_embedding      = self.embedding(torch.ones(batch_size).long().to(device=self.device))
            alpha_sum_parent    = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
            alpha_sum_completed = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
            alpha_current       = torch.zeros((batch_size, 1, height, width)).to(device=self.device)

            struct_list = []
            for bb in range(batch_size):
                struct_list.append([])

            # Track which samples have finished decoding
            finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            parent_hidden = self.init_hidden(cnn_features, images_mask)
            for i in range(num_steps):

                # word: gru-alpha
                word_hidden_first = self.word_input_gru(word_embedding, parent_hidden)

                # attention
                word_context_vec, word_alpha = self.word_attention(
                    cnn_features, word_hidden_first,
                    alpha_sum_completed, alpha_sum_parent + alpha_current, images_mask)
                
                # gru-beta
                hidden = self.word_out_gru(word_context_vec, word_hidden_first)

                # project and combine condensed image, hidden state, and previously parsed symbol
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted

                # predict next word
                word_prob = self.word_convert(word_out_state)
                _, word = word_prob.max(1)
                word_probs[:, i, :] = word_prob
                word_alphas[:, i, :, :] = word_alpha[:, 0, :, :]

                # predict struct
                struct_prob = self.struct_convert(word_out_state)
                struct_probs[:, i, :] = struct_prob
                structs = torch.sigmoid(struct_prob)

                for bb in range(batch_size):
                    # Skip samples that have already finished decoding
                    if finished[bb]:
                        continue

                    if word[bb].item() == self.STRUCT_ID: # start new struct

                        # e.g. prev: 'frac', current: 'struct'
                        # start a new sequence, so re-init alpha_current
                        alpha_struct = word_alpha[bb] + alpha_current[bb]
                        alpha_sum_parent[bb] = alpha_sum_parent[bb] + alpha_struct
                        alpha_current[bb] = torch.zeros((1, 1, height, width)).to(device=self.device)

                        for num in range(structs.shape[1]-1, -1, -1):
                            if structs[bb,num] > self.threshold:
                                struct_list[bb].append((self.struct_dict[num], hidden[bb], alpha_sum_parent[bb], alpha_struct))

                        if len(struct_list[bb]) == 0:
                            finished[bb] = True  # Mark as finished instead of break
                            continue

                        word[bb], parent_hidden[bb], alpha_sum_parent[bb], alpha_struct = struct_list[bb].pop()
                        if word[bb] == self.RIGHT_ID:
                            # completed the struct (e.g. \frac or a sup)
                            alpha_current[bb] = alpha_struct
                            alpha_sum_parent[bb] = alpha_sum_parent[bb] - alpha_struct

                        word_embedding[bb] = self.embedding(torch.LongTensor([word[bb]]).to(device=self.device))

                    elif word[bb].item() == self.EOS_ID: # end of sequece
                        if len(struct_list[bb]) == 0:
                            finished[bb] = True  # Mark as finished instead of break
                            continue

                        word[bb], parent_hidden[bb], alpha_sum_parent[bb], alpha_struct = struct_list[bb].pop()
                        word_embedding[bb] = self.embedding(torch.LongTensor([word[bb]]).to(device=self.device))
                        # completed the current sequence
                        alpha_sum_completed[bb] = alpha_sum_completed[bb] + alpha_current[bb]
                        alpha_current[bb] = torch.zeros((1, 1, height, width)).to(device=self.device)
                        if word[bb] == self.RIGHT_ID:
                            # completed the struct (e.g. \frac or a sup)
                            alpha_current[bb] = alpha_struct
                            alpha_sum_parent[bb] = alpha_sum_parent[bb] - alpha_struct


                    else: # regular word (symbol or latex command) in current sequence.
                        word_embedding[bb] = self.embedding(word[bb])
                        parent_hidden[bb] = hidden[bb].clone()
                        alpha_sum_completed[bb] = alpha_sum_completed[bb] + alpha_current[bb]
                        alpha_current[bb] = word_alpha[bb]

        return word_probs, struct_probs, word_alphas, None, c2p_probs, c2p_alphas


    def init_hidden(self, features, feature_mask):

        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)

        return torch.tanh(average)
    
    def child_to_parent(self, i, labels, cnn_features, c2p_hidden, c2p_alpha_sum_completed, c2p_alpha_sum, images_mask):
        """ child to parent """
        # Walk the hybrid tree backwards to predict parents from (child + relation)
        # E.g. for expression 2x_0: 
        # - forward:         ['2'] --> ['x', 'right']
        # - child-to-parent: ['0', 'sub'] --> 'x' 
        # So we have 2 estimations for 'x'. We use the attention vector KL and the backward/forward probabilities 
        # for x as a regulariser when calculating the loss

        batch_size, num_steps, _ = labels.shape
        height, width = cnn_features.shape[2:]
        current_type = labels[:,-(i + 1),1].clone()

        # the embedding of the last generated child (teacher forcing strategy)
        child_embedding = self.embedding(current_type)

        # the embedding of the last generated relation (also using teacher forcing strategy)
        relation = labels[:, -(i + 1), 3].clone()
        for num in range(relation.shape[0]):
            if labels[num, -(i + 1), 1] == self.STRUCT_ID: # struct
                # struct line, set parent to struct (original parent is symbol, like \frac)
                relation[num] = self.STRUCT_ID
            elif relation[num].item() not in self.struct_dict and relation[num].item() != self.EOS_ID:
                # if parent is symbol, the relation is 'right'
                relation[num] = self.RIGHT_ID
        relation_embedding = self.embedding(relation)

        c2p_alpha_struct_mask = (current_type == self.STRUCT_ID).float().view(batch_size, 1, 1, 1)
        c2p_alpha_eos_mask = (current_type != self.EOS_ID).float().view(batch_size, 1, 1, 1)

        # the partner state of reversed GRU_alpha is concatenation of (child + relation). Like ['0', 'sub'] (but then the embedding vectors)
        c2p_hidden_first = self.c2p_input_gru(torch.cat((child_embedding, relation_embedding), dim=1), c2p_hidden)

        c2p_context_vec, c2p_alpha = self.c2p_attention(
            cnn_features, c2p_hidden_first,
            c2p_alpha_sum_completed, c2p_alpha_sum, images_mask)
        
        c2p_hidden = self.c2p_out_gru(c2p_context_vec, c2p_hidden_first)

        c2p_state = self.c2p_state_weight(c2p_hidden)
        c2p_weighted_word = self.c2p_word_weight(child_embedding)
        c2p_weighted_relation = self.c2p_relation_weight(relation_embedding)
        c2p_context_weighted = self.c2p_context_weight(c2p_context_vec)

        # update word_alpha_sum for structs
        c2p_alpha_sum = c2p_alpha_sum + c2p_alpha_struct_mask * c2p_alpha
        c2p_alpha_sum_completed = c2p_alpha_sum_completed + (1-c2p_alpha_struct_mask) * c2p_alpha

        if self.params['dropout']:
            c2p_out_state = self.dropout(c2p_state + c2p_weighted_word + c2p_weighted_relation + c2p_context_weighted)
        else:
            c2p_out_state = c2p_state + c2p_weighted_word + c2p_weighted_relation + c2p_context_weighted

        return c2p_out_state, c2p_alpha, c2p_hidden, c2p_alpha_sum, c2p_alpha_sum_completed

