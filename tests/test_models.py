"""Unit tests for all model architectures — CPU only, small tensors."""

import torch
import pytest


def test_bigru_forward_shape():
    from mrm.models.bigru import BiGRUClassifier
    model = BiGRUClassifier(vocab_size=100, embedding_dim=16, hidden_dim=32, output_dim=4, n_layers=2)
    x = torch.randint(0, 100, (4, 20))  # batch=4, seq_len=20
    out = model(x)
    assert out.shape == (4, 4)


def test_bigru_load_embeddings():
    from mrm.models.bigru import BiGRUClassifier
    model = BiGRUClassifier(vocab_size=50, embedding_dim=16, output_dim=3)
    vectors = torch.randn(50, 16)
    model.load_pretrained_embeddings(vectors)
    assert torch.allclose(model.embedding.weight.data, vectors)


def test_textcnn_forward_shape():
    from mrm.models.cnn_text import TextCNNClassifier
    model = TextCNNClassifier(vocab_size=100, embedding_dim=16, n_filters=32, output_dim=5)
    x = torch.randint(0, 100, (4, 30))
    out = model(x)
    assert out.shape == (4, 5)


def test_charcnn_forward_shape():
    from mrm.models.char_cnn import CharCNNClassifier
    model = CharCNNClassifier(num_chars=68, max_seq_len=270, num_classes=3, n_conv_filters=32, n_fc_neurons=64)
    x = torch.randn(2, 68, 270)
    out = model(x)
    assert out.shape == (2, 3)


def test_text_to_onehot_shape():
    from mrm.models.char_cnn import text_to_onehot, CHAR_VOCAB
    tensor = text_to_onehot("hello world", max_len=50)
    assert tensor.shape == (len(CHAR_VOCAB), 50)
    assert tensor.sum() <= 50  # at most one 1 per position


def test_dae_reconstruction_shape():
    from mrm.models.autoencoder import DenoisingAutoencoder
    model = DenoisingAutoencoder(input_dim=64, latent_dim=8)
    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 64)


def test_vae_forward_returns_three_tensors():
    from mrm.models.autoencoder import VariationalAutoencoder
    model = VariationalAutoencoder(input_dim=64, hidden_dim=32, latent_dim=16)
    x = torch.randn(4, 64)
    recon, mu, logvar = model(x)
    assert recon.shape == (4, 64)
    assert mu.shape == (4, 16)
    assert logvar.shape == (4, 16)


def test_mage_output_shape():
    from mrm.models.mage import MAGEClassifier
    model = MAGEClassifier(embed_dim=64, num_heads=4, num_classes=3)
    embs = torch.randn(8, 64)
    out = model(embs)
    assert out.shape == (8, 3)


def test_lstm_classifier_shape():
    from mrm.models.mage import LSTMClassifier
    model = LSTMClassifier(input_dim=64, hidden_dim=32, num_classes=3)
    embs = torch.randn(4, 64)
    out = model(embs)
    assert out.shape == (4, 3)


def test_afro_xlmr_comet_student_config():
    from mrm.models.distillation import STUDENT_CONFIG
    assert STUDENT_CONFIG["num_hidden_layers"] == 6
    assert STUDENT_CONFIG["hidden_size"] == 256
    assert STUDENT_CONFIG["num_attention_heads"] == 8
    assert STUDENT_CONFIG["intermediate_size"] == 1024


def test_distillation_loss_components():
    from mrm.models.distillation import distillation_loss, AttentionProjection
    batch, heads, seq, dim = 2, 8, 10, 3
    s_logits = torch.randn(batch, dim)
    t_logits = torch.randn(batch, dim)
    s_attn = [torch.randn(batch, heads, seq, seq)]
    t_attn = [torch.randn(batch, 16, seq, seq)]  # teacher has 16 heads
    proj = AttentionProjection(seq_len=seq, student_heads=8, teacher_heads=16)
    L_total, L_dist, L_attn = distillation_loss(
        s_logits, t_logits, s_attn, t_attn, proj, temperature=2.0, alpha=0.5
    )
    assert L_total.item() >= 0
    assert L_dist.item() >= 0
    assert L_attn.item() >= 0
    # Combined loss is weighted sum
    expected = 0.5 * L_dist.item() + 0.5 * L_attn.item()
    assert abs(L_total.item() - expected) < 1e-5


def test_lida_augment_shape():
    from mrm.augmentation.lida import LiDA
    lida = LiDA(r_min=0.0, r_max=0.1)
    embs = torch.randn(10, 768)
    aug = lida.augment(embs, n_augmented=2)
    assert aug.shape == (20, 768)
