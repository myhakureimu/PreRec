import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class PWLayer(nn.Module):
    """ Single Parametric Whitening Layer """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)

class D:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def num(self):
        return 1
class UniSRec(SASRec):
    def __init__(self, truncate, H):
        
        config = {}
        config['USER_ID_FIELD'] = 'user_id'
        config['ITEM_ID_FIELD'] = 'item_id'
        config['LIST_SUFFIX'] = '_list'
        config['ITEM_LIST_LENGTH_FIELD'] = 'item_length'
        config['NEG_PREFIX'] = 'neg_'
        config['MAX_ITEM_LIST_LENGTH'] = truncate
        config['device'] = 'cuda'
        config['n_layers'] = 2
        config['n_heads'] = 2
        config['hidden_size'] = H
        config['inner_size'] = 256 # the dimensionality in feed-forward layer
        config['hidden_dropout_prob'] = 0.5
        config['attn_dropout_prob'] = 0.5
        config['hidden_act'] = 'gelu'
        config['layer_norm_eps'] = 1e-12
        config['initializer_range'] = 0.02
        config['loss_type'] = 'CE'
        
        dataset = D

        super().__init__(config, dataset)
        print('\n')
        self.train_stage = 'pretrain' #config['train_stage']
        self.temperature = 0.07 #config['temperature']
        self.lam = 0.001 #config['lambda']
        # with open('readme.txt', 'w') as f:
        #     f.write('self.train_stage: '+str(self.train_stage))
        #     f.write('self.temperature: '+str(self.temperature))
        #     f.write('self.lam: '+str(self.lam))
        
        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        
        self.moe_adaptor = MoEAdaptorLayer(
            8, #config['n_exps'],
            [768, config['hidden_size']], #config['adaptor_layers'],
            0.2, #config['adaptor_dropout_prob']
        )
        # with open('readme.txt', 'w+') as f:
        #     f.write('config[n_exps]: '+str(config['n_exps'])+'\n')
        #     f.write('config[adaptor_layers]: '+str(config['adaptor_layers'])+'\n')
        #     f.write('config[adaptor_dropout_prob]: '+str(config['adaptor_dropout_prob'])+'\n')
        
    def get_item_rec(self, item_nlp, item_debias=None, domainIdx=None, itemIdx=None):
        item_nlp = item_nlp.squeeze(1) # B, 1, F  =>  B, F
        items_emb = self.moe_adaptor(item_nlp)
        items_emb = F.normalize(items_emb, dim=1)
        items_emb = items_emb.unsqueeze(1) # B, F  =>  B, 1, F
        return items_emb
    
    def get_user_rec(self, interaction):
        item_seq = interaction['item_id_list']
        #print('item_seq: ', item_seq)
        #item_seq_len = interaction[self.ITEM_SEQ_LEN]
        #print('self.ITEM_SEQ_LEN: ',self.ITEM_SEQ_LEN)
        item_seq_len = interaction['item_length']
        #print('item_seq_len: ', item_seq_len)
        
        #print('moe_adaptor BEFORE: ', interaction['item_emb_list'].shape)
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        #print('moe_adaptor AFTER : ', item_emb_list.shape)
        
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)
        return seq_output
        
    def forward(self, item_seq, item_emb, item_seq_len):
        #print('item_seq.get_device(): ',item_seq.get_device())
        #print('item_emb.get_device(): ',item_emb.get_device())
        #print('item_seq_len.get_device(): ',item_seq_len.get_device())
        #print('item_seq: ',item_seq)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        #print('position_ids: ',position_ids)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        #print('position_ids: ',position_ids)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_items_emb = self.moe_adaptor(interaction['pos_item_emb'])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)
        
        #print(seq_output.shape)
        #print(pos_items_emb.shape)
        
        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)
        
        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, interaction):
        item_seq_aug = interaction[self.ITEM_SEQ + '_aug']
        item_seq_len_aug = interaction[self.ITEM_SEQ_LEN + '_aug']
        item_emb_list_aug = self.moe_adaptor(interaction['item_emb_list_aug'])
        seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def pretrain(self, interaction):
        #item_seq = interaction[self.ITEM_SEQ]
        #print('self.ITEM_SEQ: ',self.ITEM_SEQ)
        item_seq = interaction['item_id_list']
        #print('item_seq: ', item_seq)
        #item_seq_len = interaction[self.ITEM_SEQ_LEN]
        #print('self.ITEM_SEQ_LEN: ',self.ITEM_SEQ_LEN)
        item_seq_len = interaction['item_length']
        #print('item_seq_len: ', item_seq_len)
        #print(interaction['item_emb_list'].shape)
        
        item_emb_list = self.moe_adaptor(interaction['item_emb_list'])
        
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        #print('pos_id: ', pos_id)
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        loss_seq_item = self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
        # need interaction['pos_item_emb']
        ##### loss_seq_seq = self.seq_seq_contrastive_task(seq_output, same_pos_id, interaction)
        # need interaction['item_id_list_aug']
        # need interaction['item_length_aug']
        # need interaction['item_emb_list_aug']
        loss = loss_seq_item ##### + self.lam * loss_seq_seq
        #print(interaction)
        return loss

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)

        # Loss for fine-tuning
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.moe_adaptor(self.plm_embedding.weight)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
