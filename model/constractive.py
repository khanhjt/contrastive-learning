import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class ConstractiveProdLDA(nn.Module):
    def __init__(self,config):
        super(ConstractiveProdLDA, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.drop_lda = nn.Dropout(config.drop_lda)

        topic_mean_prior = 0.
        mean_prior = torch.Tensor(1, config.num_topics).fill_(topic_mean_prior)
        mean_prior = mean_prior.to(config.device)

        topic_mean_prior = 1 - (1. / config.num_topics)
        var_prior = torch.Tensor(1, config)



