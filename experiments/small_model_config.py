import text_model

config = text_model.Config(lstm_layers=1,
                           lstm_size=256,
                           embedding_size=64,
                           attention_num_heads=3,
                           attention_head_size=64,
                           feature_params={'vocab_size': 255},)
