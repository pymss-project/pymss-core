from pymss.modules.bandit.tfmodel import (
    ResidualRNN,
    SeqBandModellingModule as _SeqBandModellingModule,
    TimeFrequencyModellingModule,
    Transpose,
)


class SeqBandModellingModule(_SeqBandModellingModule):
    def __init__(
            self,
            n_modules: int = 12,
            emb_dim: int = 128,
            rnn_dim: int = 256,
            bidirectional: bool = True,
            rnn_type: str = "LSTM",
            parallel_mode: bool = False,
    ) -> None:
        super().__init__(
            n_modules=n_modules,
            emb_dim=emb_dim,
            rnn_dim=rnn_dim,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            parallel_mode=parallel_mode,
            sequential_transpose=not parallel_mode,
            checkpoint_segments=None if parallel_mode else n_modules,
        )


__all__ = ["ResidualRNN", "SeqBandModellingModule", "TimeFrequencyModellingModule", "Transpose"]
