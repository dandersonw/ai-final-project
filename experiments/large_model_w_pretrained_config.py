import text_model

features = {'tokens'}

config = text_model.Config(lstm_layers=2,
                           lstm_size=512,
                           embedding_size=300,
                           lstm_dropout=.5,
                           use_pretrained_embeddings=True,
                           attention_num_heads=3,
                           attention_head_size=128,
                           feature_params={'vocab_size': 255},)
