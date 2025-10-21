import torch.nn as nn
import torch
from . import encoder
from . import decoder

class Backbone(nn.Module):
    def __init__(self, params=None):
        super(Backbone, self).__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']

        if params['encoder']['net'] == 'DenseNet':
            from .encoder.densenet import DenseNet
            self.encoder = DenseNet(params=self.params)
        else:
            self.encoder = getattr(encoder, params['encoder']['net'])(params=self.params)
        
        if params['decoder']['net'] == 'SAN_decoder':
            from .decoder.decoder import SAN_decoder
            self.decoder = SAN_decoder(params=self.params)
        else:
            self.decoder = getattr(decoder, params['decoder']['net'])(params=self.params)
        self.cross = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet'][
            'conv1_stride']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):

        cnn_features = self.encoder(images)
        word_probs, struct_probs, words_alphas, struct_alphas, c2p_probs, c2p_alphas = self.decoder(cnn_features, labels, images_mask, labels_mask, is_train=is_train)

        word_average_loss = self.cross(word_probs.contiguous().reshape(-1, word_probs.shape[-1]), labels[:,:,1].reshape(-1))

        # BCEWithLogitsLoss includes sigmoid internally, so don't apply it manually
        struct_average_loss = self.bce(struct_probs, labels[:,:,4:].float())
        struct_probs = torch.sigmoid(struct_probs)  # Apply sigmoid for output/evaluation
        if labels_mask is not None:
            struct_average_loss = (struct_average_loss * labels_mask[:,:,0][:, :, None]).sum() / (labels_mask[:,:,0].sum() + 1e-10)

        if is_train:
            if self.params['decoder']['inverse']:
                parent_average_loss = self.cross(c2p_probs.contiguous().reshape(-1, word_probs.shape[-1]), labels[:, :, 3].reshape(-1))
                kl_average_loss = self.cal_kl_loss(words_alphas, c2p_alphas, labels, images_mask[:, :, ::self.ratio, ::self.ratio].contiguous(), labels_mask)
            else:
                parent_average_loss = 0
                kl_average_loss = 0

            # Calculate EOS penalty to address train/inference mismatch
            eos_penalty = self.calculate_eos_penalty(word_probs, labels, labels_mask)

            return (word_probs, struct_probs), (word_average_loss, struct_average_loss, parent_average_loss, kl_average_loss, eos_penalty)

        return (word_probs, struct_probs), (word_average_loss, struct_average_loss)

    def calculate_eos_penalty(self, word_probs, labels, labels_mask):
        """
        Calculate asymmetric penalty for EOS mispredictions.

        During training with teacher forcing, EOS is treated like any other symbol.
        However, during inference, EOS has a catastrophic effect (terminates decoding).
        This penalty explicitly addresses two critical failure modes:

        1. Premature EOS (early termination): Model predicts EOS before sequence ends
           → Missing parts of expression (most catastrophic)

        2. Missing EOS (hallucination): Model fails to predict EOS at true end
           → Attention drifts, hallucinates extra symbols

        Args:
            word_probs: [batch, time, vocab] - predicted logits
            labels: [batch, time, 11] - ground truth (labels[:,:,1] = child symbols)
            labels_mask: [batch, time, 2] - mask for valid positions

        Returns:
            eos_penalty: scalar loss
        """
        import torch.nn.functional as F

        batch_size, max_time, vocab_size = word_probs.shape
        eos_id = self.params['words'].encode(['<eos>'])[0]

        # Get softmax probabilities (not logits)
        word_probs_softmax = F.softmax(word_probs, dim=-1)
        eos_probs = word_probs_softmax[:, :, eos_id]  # [batch, time]

        # Ground truth: identify where EOS should/shouldn't appear
        true_eos = (labels[:, :, 1] == eos_id).float()  # [batch, time]
        true_non_eos = 1.0 - true_eos

        # Get valid positions from mask
        valid_mask = labels_mask[:, :, 0]  # [batch, time]

        # Case 1: Penalize predicting EOS when it shouldn't (EARLY TERMINATION)
        # This is more catastrophic - missing expression parts
        # We want eos_probs to be LOW when true_non_eos is HIGH
        early_eos_penalty = (eos_probs * true_non_eos * valid_mask).sum() / (valid_mask.sum() + 1e-10)
        early_eos_weight = self.params.get('eos_early_weight', 5.0)

        # Case 2: Penalize NOT predicting EOS when it should (HALLUCINATION)
        # We want eos_probs to be HIGH when true_eos is HIGH
        missing_eos_penalty = ((1.0 - eos_probs) * true_eos * valid_mask).sum() / (valid_mask.sum() + 1e-10)
        missing_eos_weight = self.params.get('eos_missing_weight', 3.0)

        # Combine with asymmetric weights
        eos_penalty = early_eos_weight * early_eos_penalty + missing_eos_weight * missing_eos_penalty

        return eos_penalty

    def cal_kl_loss(self, child_alphas, parent_alphas, labels, image_mask, label_mask):

        batch_size, steps, height, width = child_alphas.shape
        new_child_alphas = torch.zeros((batch_size, steps, height, width)).to(self.params['device'])
        new_child_alphas[:, 1:, :, :] = child_alphas[:,:-1,:,:].clone()
        new_child_alphas = new_child_alphas.reshape((batch_size*steps, height, width))
        parent_ids = labels[:,:,2] + steps * torch.arange(batch_size)[:,None].to(self.params['device'])

        new_child_alphas = new_child_alphas[parent_ids].contiguous()
        new_child_alphas = new_child_alphas.reshape((batch_size, steps, height, width))[:, 1:, :, :].contiguous()
        new_parent_alphas = parent_alphas[:,1:,:,:].contiguous()

        KL_alpha = new_child_alphas * (torch.log(new_child_alphas + 1e-10) - torch.log(new_parent_alphas + 1e-10)) * image_mask
        KL_loss = (KL_alpha.sum(-1).sum(-1) * label_mask[:,:-1, 0]).sum(-1).sum(-1) / (label_mask.sum() - batch_size)

        return KL_loss

