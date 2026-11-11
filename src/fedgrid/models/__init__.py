"""Classifier zoo: CNN-LSTM reference, spiking (LIF), and selective SSM classifiers.

All three consume the data contract's ``(B, C=11, T=64)`` float32 windows and
emit ``(B, 8)`` logits. The SNN is pure PyTorch (no snntorch) so SynOps counting
and surrogate gradients stay auditable; the SSM is a pure-PyTorch Mamba/S4D-style
selective state-space model (no mamba-ssm dependency), the third efficiency-frame
classifier.
"""

from __future__ import annotations

from fedgrid.models.cnn_lstm import CNNLSTM
from fedgrid.models.snn import LIF, SpikingClassifier
from fedgrid.models.ssm import SSMBlock, SSMClassifier

__all__ = ["CNNLSTM", "LIF", "SSMBlock", "SSMClassifier", "SpikingClassifier"]
