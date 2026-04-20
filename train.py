"""
PyTorch Lightning training script for music2latent (Consistency Autoencoder).

Based on the ISMIR 2024 paper: https://arxiv.org/abs/2408.06500

The model is a Consistency Autoencoder:
  - Encoder: real/imag STFT → latent representation (~10 Hz, 64 ch)
  - Decoder: latents → pyramid skip features for UNet
  - UNet: consistency model that denoises audio conditioned on latents

Training uses consistency training (self-distillation with EMA teacher).
No pre-trained diffusion model is needed — the EMA copy of the student
acts as the teacher, and the consistency property emerges from training.

Usage:
  python train.py --data_dir /path/to/audio
  python train.py --data_dir /path/to/audio --max_epochs 1000 --precision bf16-mixed
"""

import os
import math
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from music2latent.models import UNet, Encoder, Decoder
from musicsep_visualizer import VisualizationHook


def compute_sdr(clean, recon):
    """Fast waveform SDR (SI-SDR style), pure GPU tensor ops."""
    noise = clean - recon
    return 10.0 * torch.log10(
        torch.clamp((clean ** 2).sum(dim=-1) / (noise ** 2).sum(dim=-1), min=1e-8)
    )  # [B]
from music2latent.audio import to_representation_encoder, to_waveform
from music2latent.utils import get_c, get_sigma, add_noise
from music2latent.hparams import (
    data_channels,
    hop,
    sigma_min,
    sigma_max,
    rho,
    sigma_data,
    mixed_precision,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Loads audio files → real/imag STFT chunks of fixed length.

    Time dimension must be divisible by 16 (4 downsampling stages).
    For hop=512, 4 seconds → 345 frames → padded to 352 (352/16 = 22).
    """

    EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff"}

    def __init__(
        self,
        root_dir: str,
        sample_rate: int = 44100,
        segment_seconds: float = 4.0,
        augment: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.augment = augment

        # Compute STFT dimensions
        self.segment_samples = int(segment_seconds * sample_rate)
        n_fft = hop * 4  # = 2048, window size
        self.freq_bins = hop * 2  # = 1024

        # Time frames from STFT (center=True padding adds n_fft/2 on each side)
        # We'll compute the actual time dim from the audio after STFT and pad
        self.time_divisor = 16  # must be divisible by this

        # Collect audio files
        self.files = sorted(
            str(p) for ext in self.EXTENSIONS
            for p in Path(root_dir).rglob(f"*{ext}")
        )
        if not self.files:
            raise ValueError(f"No audio files found in {root_dir}")

        # Index: (file_idx, segment_idx_within_file)
        self.segments = []
        for fi, fp in enumerate(self.files):
            try:
                info = sf.info(fp)
                dur = info.duration
                n_segs = max(1, int(dur / segment_seconds))
                for si in range(n_segs):
                    self.segments.append((fi, si))
            except Exception:
                continue

    def __len__(self):
        return len(self.segments)

    def _load_audio(self, file_idx: int):
        """Load full audio file, return numpy array [channels, samples]."""
        audio, sr = sf.read(self.files[file_idx], dtype="float32", always_2d=True)
        audio = audio.T  # [channels, samples]

        if sr != self.sample_rate:
            import torchaudio
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, self.sample_rate
            ).numpy()

        # Ensure 2 channels
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        audio = audio[:2]

        return audio

    def __getitem__(self, idx):
        file_idx, seg_idx = self.segments[idx]
        audio = self._load_audio(file_idx)  # [2, samples]

        # Random crop within file (use seg_idx as offset seed + random jitter)
        length = audio.shape[-1]
        if length > self.segment_samples:
            max_offset = length - self.segment_samples
            base_offset = (seg_idx * self.segment_samples) % max_offset
            if self.augment:
                jitter = np.random.randint(-self.segment_samples // 4, self.segment_samples // 4)
                start = (base_offset + jitter) % max_offset
            else:
                start = base_offset
            audio = audio[:, start : start + self.segment_samples]
        elif length < self.segment_samples:
            audio = np.pad(audio, ((0, 0), (0, self.segment_samples - length)))

        # Augmentation
        if self.augment:
            audio = audio * np.random.uniform(0.7, 1.0)
            if np.random.random() < 0.5:
                audio = -audio

        # Convert to STFT representation
        # to_representation_encoder expects [channels, samples] -> [channels, 2, 1024, T]
        # where dim0 = audio channels (L/R), dim1 = real/imag
        audio_t = torch.from_numpy(audio).float()  # [2, samples]
        repr = to_representation_encoder(audio_t)   # [2, 2, 1024, T]

        # Pick a random audio channel (L or R) — model processes one channel at a time
        ch = np.random.randint(0, 2) if self.augment else 0
        repr = repr[ch]  # [2, 1024, T]

        # Pad time dim to be divisible by 16
        T = repr.shape[-1]
        pad_T = (self.time_divisor - (T % self.time_divisor)) % self.time_divisor
        if pad_T > 0:
            repr = F.pad(repr, (0, pad_T))

        return repr


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class Music2LatentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        batch_size: int = 1,
        num_workers: int = 4,
        segment_seconds: float = 4.0,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir or train_dir  # fallback to train if no val split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.segment_seconds = segment_seconds
        self.sample_rate = sample_rate

    def setup(self, stage=None):
        self.train_dataset = AudioDataset(
            self.train_dir, self.sample_rate, self.segment_seconds, augment=True,
        )
        self.val_dataset = AudioDataset(
            self.val_dir, self.sample_rate, self.segment_seconds, augment=False,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.998):
        self.decay = decay
        self.params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.params

    def load_state_dict(self, state):
        self.params = state


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class ConsistencyAutoencoder(pl.LightningModule):
    """
    Consistency Training for the music2latent autoencoder.

    The model learns to map any noise level directly to clean data in one step,
    conditioned on encoder latents. The teacher is an EMA copy of the student.

    Loss: || F_θ(x + σ·ε) - F_θ⁻(x + σ'·ε') ||²
    where σ' < σ are adjacent noise levels and θ⁻ is the EMA teacher.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        ema_decay: float = 0.998,
        num_noise_levels: int = 64,
        warmup_steps: int = 5000,
        P_mean: float = -1.1,
        P_std: float = 1.6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.num_noise_levels = num_noise_levels
        self.warmup_steps = warmup_steps
        self.P_mean = P_mean
        self.P_std = P_std

        # Main model (student)
        self.model = UNet()

        # EMA teacher (initialized on_fit_start after device placement)
        self.ema = None
        self.ema_model = None

        # Precompute discretized sigma schedule: σ_1 ... σ_{K+1}
        # σ_1 = sigma_min, σ_{K+1} = sigma_max
        self.register_buffer(
            "sigmas",
            torch.tensor([
                get_sigma(i, num_noise_levels + 1)
                for i in range(1, num_noise_levels + 2)
            ]),
        )

    def on_fit_start(self):
        """Create EMA teacher and visualization hooks after model is on device."""
        if self.ema is None:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema_model = UNet().to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            self.ema_model.eval()
            for p in self.ema_model.parameters():
                p.requires_grad = False

        # Visualization hooks (nn.Identity passthrough — only active on CUDA)
        self.input_viz = VisualizationHook("train/recon_waveform")
        self.target_viz = VisualizationHook("train/input_waveform")
        self.student_stft_viz = VisualizationHook("train/student_stft_real")
        self.target_stft_viz = VisualizationHook("train/input_stft_real")
        self.val_input_viz = VisualizationHook("val/recon_waveform")
        self.val_target_viz = VisualizationHook("val/input_waveform")
        self.val_recon_stft_viz = VisualizationHook("val/recon_stft_real")
        self.val_target_stft_viz = VisualizationHook("val/input_stft_real")

    def forward(self, x_0, sigma=None):
        """Encode then decode at given noise level."""
        latents = self.model.encoder(x_0)
        if sigma is None:
            sigma = sigma_min
        return self.model(latents, x_0, sigma=sigma)

    def _sync_ema_model(self):
        """Copy EMA shadow params into the teacher model."""
        if self.ema is not None and self.ema_model is not None:
            for n, p in self.ema_model.named_parameters():
                if n in self.ema.params:
                    p.data.copy_(self.ema.params[n])

    def training_step(self, batch, batch_idx):
        x_0 = batch  # [B, 2, 1024, T] — real/imag STFT, T divisible by 16
        B = x_0.shape[0]
        device = x_0.device

        # ---- Consistency Training Loss ----
        # Sample σ from log-normal distribution (Karras et al. EDM2 parameterization)
        rnd_normal = torch.randn(B, device=device)
        log_sigma = self.P_mean + self.P_std * rnd_normal
        sigma_next = log_sigma.exp().clamp(sigma_min, sigma_max)  # σ_{n+1} (higher)

        # For each σ_{n+1}, find the adjacent σ_n (lower) in the discretized schedule
        # σ_n is the largest schedule value that is < σ_{n+1}
        sigma_n = torch.zeros_like(sigma_next)
        for i in range(B):
            s = sigma_next[i].item()
            # Find largest schedule sigma < s
            mask = self.sigmas < s
            if mask.any():
                sigma_n[i] = self.sigmas[mask].max()
            else:
                sigma_n[i] = self.sigmas[0]  # sigma_min

        # Create noisy inputs
        noise = torch.randn_like(x_0)

        # x at σ_{n+1} (higher noise — teacher input)
        x_noisy_next = x_0 + sigma_next.view(-1, 1, 1, 1) * noise

        # x at σ_n (lower noise — from same noise realization, scaled)
        # x_n = x_0 + σ_n/σ_{n+1} · (x_{n+1} - x_0)
        x_noisy_n = x_0 + (sigma_n / sigma_next).view(-1, 1, 1, 1) * (x_noisy_next - x_0)

        # Teacher prediction at σ_n (lower noise level)
        with torch.no_grad():
            self._sync_ema_model()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
                teacher_target = self.ema_model(
                    self.ema_model.encoder(x_0),  # teacher encodes clean x_0
                    x_noisy_n,
                    sigma=sigma_n,  # per-sample sigma
                )

        # Student prediction at σ_{n+1} (higher noise level)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=mixed_precision):
            student_pred = self.model(
                self.model.encoder(x_0),  # student encodes clean x_0
                x_noisy_next,
                sigma=sigma_next,  # per-sample sigma
            )

        # Huber loss (more robust than MSE for consistency training)
        loss = F.smooth_l1_loss(student_pred.float(), teacher_target.float())

        # P-weighting (Karras et al.) — upweights mid-range sigmas
        p_weight = (sigma_next ** 2 + sigma_data ** 2) / (sigma_next * sigma_data) ** 2
        loss = (loss * p_weight).mean()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/sigma_mean", sigma_next.mean(), on_step=True, on_epoch=False)

        # Waveform SDR of student reconstruction vs clean target
        with torch.no_grad():
            recon_wave = to_waveform(student_pred.detach().float())
            clean_wave = to_waveform(x_0.detach().float())
            sdr = compute_sdr(clean_wave, recon_wave)
            self.log("train/sdr", sdr.mean(), prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

            # ---- Visualization ----
            # VisualizationHook is a zero-overhead nn.Identity until the
            # subprocess starts; it rate-limits internally to ~30 FPS.
            if x_0.is_cuda:
                self.input_viz(recon_wave[:1])
                self.target_viz(clean_wave[:1])
                self.student_stft_viz(student_pred[0, 0].detach().float())
                self.target_stft_viz(x_0[0, 0].detach().float())

        return loss

    def validation_step(self, batch, batch_idx):
        x_0 = batch

        self._sync_ema_model()

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=mixed_precision):
                # Reconstruction at σ_min (near-identity, but measures encoder quality)
                recon = self.forward(x_0, sigma=sigma_min)
                recon_loss = F.mse_loss(recon.float(), x_0.float())

                # Reconstruction at a moderate noise level (σ from schedule)
                # This tests whether the model can reconstruct from noisy input
                sigma_test = get_sigma(10, self.num_noise_levels + 1)
                noise = torch.randn_like(x_0)
                x_noisy = x_0 + sigma_test * noise
                latents = self.model.encoder(x_0)
                recon_noisy = self.model(latents, x_noisy, sigma=sigma_test)
                noisy_recon_loss = F.mse_loss(recon_noisy.float(), x_0.float())

            # ---- Visualization ----
            if x_0.is_cuda:
                recon_wave = to_waveform(recon.float())
                clean_wave = to_waveform(x_0.float())
                self.val_input_viz(recon_wave[:1])
                self.val_target_viz(clean_wave[:1])
                self.val_recon_stft_viz(recon[0, 0].float())
                self.val_target_stft_viz(x_0[0, 0].float())

        self.log("val/recon_loss", recon_loss, prog_bar=True, sync_dist=True)
        self.log("val/noisy_recon_loss", noisy_recon_loss, sync_dist=True)

        return recon_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update EMA after each optimizer step."""
        if self.ema is not None:
            self.ema.update(self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-8,
        )

        # Warmup then cosine decay
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps - self.warmup_steps if self.trainer.max_steps else 100000,
            eta_min=self.learning_rate * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_params"] = {k: v.cpu() for k, v in self.ema.state_dict().items()}

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and "ema_params" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_params"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config_defaults(parser: argparse.ArgumentParser, config_path: str | None):
    """Populate parser defaults from a TOML config file (without parsing CLI yet)."""
    if config_path is None:
        return
    import tomllib
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    for action in parser._actions:
        if action.dest in ("help", "config"):
            continue
        if action.dest in cfg:
            action.default = cfg[action.dest]


def main():
    parser = argparse.ArgumentParser(description="Train music2latent Consistency Autoencoder")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to TOML config file (values used as defaults, CLI overrides)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir with train/ and test/ subdirectories of audio files")
    parser.add_argument("--segment_seconds", type=float, default=4.0,
                        help="Audio segment length in seconds (default: 4.0)")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (2 fits ~15GB on RTX 3090, 1 ~8GB)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (overrides max_epochs)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.998)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = bs × this)")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume training from")
    parser.add_argument("--num_noise_levels", type=int, default=64)
    parser.add_argument("--P_mean", type=float, default=-1.1,
                        help="Mean of log-sigma sampling distribution")
    parser.add_argument("--P_std", type=float, default=1.6,
                        help="Std of log-sigma sampling distribution")
    # Pre-parse just --config to load defaults, then do the full parse
    _pre = parser.parse_known_args()[0]
    _load_config_defaults(parser, _pre.config)
    args = parser.parse_args()

    # Data
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(train_dir):
        train_dir = args.data_dir
    if not os.path.isdir(val_dir):
        val_dir = None  # will fall back to train_dir

    dm = Music2LatentDataModule(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_seconds=args.segment_seconds,
        sample_rate=args.sample_rate,
    )

    # Model
    model = ConsistencyAutoencoder(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
        num_noise_levels=args.num_noise_levels,
        warmup_steps=args.warmup_steps,
        P_mean=args.P_mean,
        P_std=args.P_std,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="music2latent-{epoch:04d}-{val/recon_loss:.6f}",
            monitor="val/recon_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=10,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = TensorBoardLogger(save_dir=args.log_dir, name="music2latent")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        benchmark=True,
    )

    trainer.fit(model, dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
