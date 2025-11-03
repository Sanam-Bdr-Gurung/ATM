# models/of_inference.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch, torchaudio
import torch.nn.functional as F

from models.of_model import OnsetsAndFrames
from models.notes_interface import NotesTranscriber, midi_to_hz

class OFTranscriber(NotesTranscriber):
    def __init__(
        self,
        device: str = "cpu",
        sr_model: int = 22050,
        n_mels: int = 229,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 30.0,
        fmax: float = 8000.0,
        n_pitches: int = 88,      # MIDI 21..108 (A0..C8) by default
        midi_low: int = 21,
        midi_high: int = 108,
        onset_thresh: float = 0.5,
        frame_thresh: float = 0.5,
        checkpoint_path: Optional[str] = None
    ):
        self.device = torch.device(device)
        self.sr_model = sr_model
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.midi_low = midi_low
        self.midi_high = midi_high
        self.n_pitches = n_pitches
        self.onset_thresh = onset_thresh
        self.frame_thresh = frame_thresh

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr_model, n_fft=n_fft, hop_length=hop_length,
            f_min=fmin, f_max=fmax, n_mels=n_mels, center=True, power=2.0
        ).to(self.device)
        self.amplog = torch.log

        self.model = OnsetsAndFrames(n_mels=n_mels, hidden=128, gru_layers=2, n_pitches=n_pitches).to(self.device)
        self.model.eval()

        if checkpoint_path:
            try:
                if checkpoint_path.endswith(".safetensors"):
                    try:
                        from safetensors.torch import load_file as safe_load
                    except ImportError as _:
                        raise RuntimeError(
                            "Got a .safetensors checkpoint but safetensors is not installed. "
                            "Run: pip install safetensors"
                        )
                    state = safe_load(checkpoint_path, device=self.device)
                else:
                    ckpt = torch.load(checkpoint_path, map_location=self.device)
                    state = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state, strict=False)
            except Exception as e:
                # Don't kill the server—log and continue with random weights
                import traceback
                print("[WARN] Failed to load checkpoint:", repr(e))
                traceback.print_exc()

    def _prep_audio(self, y: np.ndarray, sr: int) -> torch.Tensor:
        x = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,T]
        if sr != self.sr_model:
            x = torchaudio.functional.resample(x, sr, self.sr_model)
        return x

    def _audio_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, T]
        S = self.melspec(x) + 1e-10              # [1, F, T]
        logS = self.amplog(S)                    # ln power
        logS = (logS - logS.mean()) / (logS.std() + 1e-8)  # simple norm
        logS = logS.unsqueeze(1)                 # [1, 1, F, T]
        return logS

    def _postprocess(self, onset_logits: torch.Tensor, frame_logits: torch.Tensor, sr: int) -> List[Dict]:
        """
        Turn per-time pitch logits into note events by thresholding + merging.
        """
        onset_prob = torch.sigmoid(onset_logits)[0].cpu().numpy()  # [T,P]
        frame_prob = torch.sigmoid(frame_logits)[0].cpu().numpy()  # [T,P]
        T, P = onset_prob.shape

        # Binarize
        onset_mask = (onset_prob >= self.onset_thresh).astype(np.uint8)
        frame_mask = (frame_prob >= self.frame_thresh).astype(np.uint8)

        # Simple event extraction
        hop_t = self.hop_length / float(self.sr_model)
        events: List[Dict] = []
        active = {}

        for t in range(T):
            # pitches are 0..P-1 -> map to midi
            for p in range(P):
                midi = self.midi_low + p
                if onset_mask[t, p] == 1:
                    # start new or restart
                    active[midi] = t
                elif frame_mask[t, p] == 0:
                    # if inactive, ignore
                    if midi in active:
                        t_on = active.pop(midi)
                        t_off = t
                        if t_off > t_on:
                            e = {
                                "t_on": round(t_on * hop_t, 4),
                                "t_off": round(t_off * hop_t, 4),
                                "midi": int(midi),
                                "freq_hz": float(midi_to_hz(midi)),
                                "conf": 1.0  # placeholder; could use mean prob
                            }
                            events.append(e)
            # sustain notes with frame_mask=1 continue automatically

        # close leftovers
        if active:
            t_last = T-1
            for midi, t_on in list(active.items()):
                if t_last > t_on:
                    events.append({
                        "t_on": round(t_on * hop_t, 4),
                        "t_off": round(t_last * hop_t, 4),
                        "midi": int(midi),
                        "freq_hz": float(midi_to_hz(midi)),
                        "conf": 1.0
                    })

        # sort by onset
        events.sort(key=lambda e: e["t_on"])
        return events

    @torch.no_grad()
    def transcribe(self, y: np.ndarray, sr: int) -> List[Dict]:
        x = self._prep_audio(y, sr)       # [1,T']
        mel = self._audio_to_mel(x)       # [1,1,F,Tm]
        onset_logits, frame_logits = self.model(mel)  # [1,Tm,P], [1,Tm,P]
        return self._postprocess(onset_logits, frame_logits, sr=self.sr_model)
