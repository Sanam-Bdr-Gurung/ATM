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


    @torch.no_grad()
    def transcribe_chunked(self,
                        y: np.ndarray, sr: int,
                        chunk_sec: float = 2.0,
                        hop_sec: float = 1.0,
                        onset_filt: int = 3,
                        frame_filt: int = 5,
                        th_on_hi: float = 0.5,
                        th_on_lo: float = 0.3,
                        th_fr: float = 0.5) -> list:
        """
        Pseudo-streaming inference with overlap-add of probabilities.
        The model is BiGRU (non-causal), so this is for latency smoothing,
        not strict causality. Good enough to prepare for real streaming later.
        """
        MIN_L = 4  # minimum time frames to run the model
        x = self._prep_audio(y, sr)  # [1, T]
        
        if x.shape[-1] == 0:
            return []
         
        # compute full mel once; then slice on time axis
        mel = self._audio_to_mel(x)  # [1,1,F,Tm]
        if mel.shape[-1] < MIN_L:
            # fallback to full (or just return empty list)
            onset_logits, frame_logits = self.model(mel)
            hop_t = self.hop_length / float(self.sr_model)
            return _logit_to_events(onset_logits[0], frame_logits[0],
                                    hop_t, self.midi_low,
                                    onset_filt, frame_filt, th_on_hi, th_on_lo, th_fr)
        _, _, F, Tm = mel.shape
        chunk = max(1, int(round((chunk_sec * self.sr_model) / self.hop_length)))  # in mel frames
        step  = max(1, int(round((hop_sec   * self.sr_model) / self.hop_length)))  # in mel frames
        overlap = max(0, chunk - step)
        if chunk <= 1 or step <= 0:
            # fallback to full
            onset_logits, frame_logits = self.model(mel)
            hop_t = self.hop_length / float(self.sr_model)
            return _logit_to_events(onset_logits[0], frame_logits[0], hop_t,
                                    self.midi_low, onset_filt, frame_filt, th_on_hi, th_on_lo, th_fr)

        onset_chunks, frame_chunks, lengths = [], [], []
       
        for t0 in range(0, Tm, step):
            t1 = min(Tm, t0 + chunk)
            m_slice = mel[:, :, :, t0:t1]  # [1,1,F,L]
            if m_slice.shape[-1] < MIN_L:
                continue  # skip tiny tail slice
            on, fr = self.model(m_slice)   # [1,L,P]
            onset_chunks.append(on[0].cpu().numpy())
            frame_chunks.append(fr[0].cpu().numpy())
            lengths.append(m_slice.shape[-1])

        if not onset_chunks:
            return []

        # stitch with overlap-add averaging
        P = onset_chunks[0].shape[1]
        T_full = Tm
        onset_full = _overlap_add_probs(onset_chunks, lengths, T_full, overlap)
        frame_full = _overlap_add_probs(frame_chunks, lengths, T_full, overlap)

        hop_t = self.hop_length / float(self.sr_model)
        return _logit_to_events(torch.from_numpy(onset_full),
                                torch.from_numpy(frame_full),
                                hop_t, self.midi_low,
                                onset_filt, frame_filt, th_on_hi, th_on_lo, th_fr)


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x
    k = int(k) if int(k) % 2 == 1 else int(k) + 1  # force odd
    from collections import deque
    # simple per-column filter; for probabilities we can use numpy's pad+sliding
    import numpy as _np
    pad = k // 2
    xp = _np.pad(x, ((pad, pad), (0,0)), mode="edge")
    out = _np.empty_like(x)
    for i in range(x.shape[0]):
        window = xp[i:i+k]
        out[i] = _np.median(window, axis=0)
    return out

def _overlap_add_probs(chunks, lengths, T_full, overlap):
    """
    chunks: list of np arrays [T_chunk, P]
    lengths: list of int effective lengths (exclude right padding)
    T_full: total frames in concatenated timeline
    overlap: number of frames overlapped between chunks
    """
    P = chunks[0].shape[1]
    acc = np.zeros((T_full, P), dtype=np.float32)
    wgt = np.zeros((T_full, P), dtype=np.float32)

    t = 0
    for arr, L in zip(chunks, lengths):
        # arr shape [T_chunk, P]
        end = min(t + L, T_full)
        seg = arr[: (end - t)]
        acc[t:end] += seg
        wgt[t:end] += 1.0
        # advance by (L - overlap) frames
        t += max(1, L - overlap)

    wgt[wgt == 0] = 1.0
    return acc / wgt

def _hysteresis(onset_prob: np.ndarray, frame_prob: np.ndarray,
                th_on_hi=0.5, th_on_lo=0.3, th_fr=0.5):
    """
    onset_prob, frame_prob: [T, P] in [0,1]
    Returns onset_mask [T,P], frame_mask [T,P].
    Onset hysteresis reduces spurious starts.
    """
    T, P = onset_prob.shape
    onset_mask = np.zeros((T,P), dtype=np.uint8)
    was_high = np.zeros(P, dtype=np.uint8)
    for t in range(T):
        hi = onset_prob[t] >= th_on_hi
        lo = onset_prob[t] >= th_on_lo
        fire = (hi | (was_high & lo))
        onset_mask[t] = fire.astype(np.uint8)
        was_high = fire.astype(np.uint8)

    frame_mask = (frame_prob >= th_fr).astype(np.uint8)
    return onset_mask, frame_mask

def _logit_to_events(onset_logits, frame_logits, hop_t, midi_low, onset_filt=0, frame_filt=0,
                     th_on_hi=0.5, th_on_lo=0.3, th_fr=0.5):
    onset_prob = torch.sigmoid(onset_logits).cpu().numpy()
    frame_prob = torch.sigmoid(frame_logits).cpu().numpy()

    if onset_filt > 1:
        onset_prob = _median_filter_1d(onset_prob, onset_filt)
    if frame_filt > 1:
        frame_prob = _median_filter_1d(frame_prob, frame_filt)

    onset_mask, frame_mask = _hysteresis(onset_prob, frame_prob, th_on_hi, th_on_lo, th_fr)

    T, P = onset_mask.shape
    events = []
    active = {}

    for t in range(T):
        for p in range(P):
            midi = midi_low + p
            if onset_mask[t, p]:
                active[midi] = t
            elif frame_mask[t, p] == 0:
                if midi in active:
                    t_on = active.pop(midi)
                    if t > t_on:
                        events.append({
                            "t_on": round(t_on * hop_t, 4),
                            "t_off": round(t * hop_t, 4),
                            "midi": int(midi),
                            "freq_hz": float(440.0 * (2.0 ** ((midi - 69)/12.0))),
                            "conf": 1.0
                        })
    # close leftovers
    if active:
        t_last = T - 1
        for m, t_on in active.items():
            if t_last > t_on:
                events.append({
                    "t_on": round(t_on * hop_t, 4),
                    "t_off": round(t_last * hop_t, 4),
                    "midi": int(m),
                    "freq_hz": float(440.0 * (2.0 ** ((m - 69)/12.0))),
                    "conf": 1.0
                })
    events.sort(key=lambda e: e["t_on"])
    return events
