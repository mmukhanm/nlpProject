"""Проверка кода на временной случайной tiny-модели, НЕ результат pretrained BERT."""
import numpy as np
import pandas as pd
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from torch.utils.data import DataLoader
from nlp_project.transformer import ReviewDataset, encode, fine_tune, probabilities


def test_encoder_training_and_checkpoint(tmp_path):
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'good', 'bad', 'phone', 'not', 'great', 'battery']
    source = tmp_path / 'source'; source.mkdir()
    (source / 'vocab.txt').write_text('\n'.join(vocab) + '\n')
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(vocab)}, do_lower_case=True)
    # Recent Transformers accepts vocab=; this assertion catches an incompatible API.
    assert tokenizer.convert_tokens_to_ids('phone') != tokenizer.unk_token_id
    tokenizer.save_pretrained(source)
    config = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)
    encoder = BertModel(config)
    encoder.save_pretrained(source)
    frame = pd.DataFrame({'text': ['good phone', 'bad battery', 'great phone', 'not good'], 'label': [1, 0, 1, 0]})
    z = encode(encoder, tokenizer, frame.text.tolist())
    assert z.shape == (4, 16) and np.isfinite(z).all()
    model, history = fine_tune(source, tokenizer, frame, frame, 1e-3, 1., 'unit_random_only', tmp_path)
    assert len(history) == 4 and np.isfinite([r['train_loss'] for r in history]).all()
    prob = probabilities(model, DataLoader(ReviewDataset(frame, tokenizer), batch_size=2))
    assert prob.shape == (4,) and ((prob >= 0) & (prob <= 1)).all()
    assert (tmp_path / 'unit_random_only/training_config.json').exists()
