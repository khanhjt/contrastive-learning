import torch 
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(config.num_topics, config.vocab_size)
        self.fc.apply(self.xavier_weight_init)

    def xavier_weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            init.zeros_(m.bias)

    def forword(self, z):
        x_recon = F.softmax(self.fc(z), dim=1)
        return x_recon