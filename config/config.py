class Config:
    hidden_size = 200
    num_topics = 100
    drop_lda = 0.2
    vocab_size = 31874
    k = 100
    learn_prior = False
    device = 'cuda'
    batch_size = 16
    lr = 5e-3
    epochs = 500
    train_cl = True