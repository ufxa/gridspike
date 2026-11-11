"""CNN-LSTM reference classifier with an analytic dense-MAC count.

The MAC count is the denominator of the paper's headline SNN-energy ratio, so it
is computed analytically from the architecture, never profiled.
"""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["WINDOW_T", "CNNLSTM"]

# Data-contract window length (CONTRACT.md §Data contract: T = 64). The analytic
# MAC count is defined for one contract window; conv/LSTM layers themselves are
# length-agnostic at runtime.
WINDOW_T: int = 64


class CNNLSTM(nn.Module):
    """1-D conv front-end + LSTM temporal head over ``(B, C, T)`` windows.

    Each conv block is Conv1d(k=5, same padding) -> BatchNorm1d -> ReLU ->
    MaxPool1d(2), so the temporal axis halves per block (64 -> 32 -> 16 with the
    default two blocks). The LSTM consumes the pooled sequence time-major and the
    classification head reads the final hidden state — the whole window must be
    seen before a verdict, matching how a merging-unit buffer is classified.
    """

    def __init__(
        self,
        in_ch: int = 11,
        n_classes: int = 8,
        conv_channels: tuple[int, ...] = (32, 64),
        kernel: int = 5,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.conv_channels = tuple(conv_channels)
        self.kernel = kernel
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        blocks: list[nn.Module] = []
        prev = in_ch
        for ch in self.conv_channels:
            blocks += [
                # Same padding keeps the analytic MAC count a clean T_in per tap.
                nn.Conv1d(prev, ch, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            prev = ch
        self.conv = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            input_size=prev,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, T) -> (B, n_classes)`` logits."""
        z = self.conv(x)  # (B, C_last, T')
        z = z.permute(0, 2, 1)  # LSTM wants (B, T', C_last)
        out, _ = self.lstm(z)
        # Final hidden state = summary after the full window has been observed.
        return self.head(self.dropout(out[:, -1, :]))

    def macs_per_inference(self) -> int:
        """Analytic dense multiply-accumulate count for one contract ``(C, T=64)`` window.

        Counts multiplies in conv kernels, LSTM gate matmuls and the linear head.
        Excluded by the standard convention (Horowitz-style accounting counts
        arithmetic dominated by MACs): biases, BatchNorm (an affine folded into
        the conv at inference), activations, pooling and element-wise LSTM gate
        arithmetic — all sub-percent of the total here.

        Default config, hand check:
          conv1: 64*32*11*5   = 112,640     (T_in=64, same padding)
          conv2: 32*64*32*5   = 327,680     (T halved by pool -> 32)
          lstm : 16*4*128*(64+128) = 1,572,864   (T'=16; 4 gates, W_ih + W_hh)
          head : 128*8        = 1,024
          total               = 2,014,208
        """
        macs = 0
        t = WINDOW_T
        prev = self.in_ch
        for ch in self.conv_channels:
            macs += t * ch * prev * self.kernel  # same padding: T_out == T_in
            t //= 2  # MaxPool1d(2) floors, matching nn.MaxPool1d
            prev = ch
        for layer in range(self.lstm_layers):
            in_size = prev if layer == 0 else self.lstm_hidden
            # Four gates, each an (in + hidden) -> hidden matmul, per timestep.
            macs += t * 4 * self.lstm_hidden * (in_size + self.lstm_hidden)
        macs += self.lstm_hidden * self.n_classes
        return macs
