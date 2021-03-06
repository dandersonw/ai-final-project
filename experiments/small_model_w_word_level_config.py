import text_model

features = {'tokens', 'word_tokens', 'uncased_word_tokens'}

config = text_model.Config(lstm_size=256,
                           embedding_size=64,
                           use_word_level_embeddings=True,
                           dense_regularization_coef=1e-4,
                           dense_dropout=.5,
                           lstm_dropout=.5,
                           attention_num_heads=3,
                           attention_head_size=64,
                           feature_params={'vocab_size': 255},)
