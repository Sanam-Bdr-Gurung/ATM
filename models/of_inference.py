# models/of_inference.py
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch, torchaudio
import torch.nn.functional as F
from onsets_and_frames.constants import N_MELS as OF_N_MELS
from onsets_and_frames.transcriber import OnsetsAndFrames as OFOrig
from models.notes_interface import NotesTranscriber, midi_to_hz
import onsets_and_frames.mel as of_mel



# piano range A0–C8 used in Onsets & Frames
OF_N_PITCHES = 88
OF_MIN_MIDI = 21
OF_MAX_MIDI = OF_MIN_MIDI + OF_N_PITCHES - 1

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
        midi_low: int = 21,  # <-- A0
        midi_high: int = 108,# <-- C8 (inclusive)
        onset_thresh: float = 0.5,
        frame_thresh: float = 0.5,
        checkpoint_path: Optional[str] = None
    ):
        self.device = torch.device(device)
        self.sr_model = sr_model
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_pitches = OF_N_PITCHES
        self.midi_low = OF_MIN_MIDI
        self.midi_high = OF_MAX_MIDI
        self.onset_thresh = onset_thresh
        self.frame_thresh = frame_thresh

            # Use original Onsets & Frames mel front-end
        of_mel.melspectrogram.to(self.device)
        self.of_melspec = of_mel.melspectrogram

        self.model = OFOrig( input_features=OF_N_MELS,output_features=OF_N_PITCHES, model_complexity=48,).to(self.device)
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
                    # We still need an OFOrig() instance in this case
                    self.model = OFOrig(OF_N_MELS, OF_N_PITCHES).to(self.device)
                    fixed = {k.replace("module.", ""): v for k, v in state.items()}
                    missing, unexpected = self.model.load_state_dict(fixed, strict=False)
                    print("[INFO] Loaded safetensors checkpoint with strict=False")
                    if missing:
                        print("[WARN] Missing keys:", missing[:10], "...")
                    if unexpected:
                        print("[WARN] Unexpected keys:", unexpected[:10], "...")
                else:
                    import torch.nn as nn
                    # IMPORTANT: weights_only=False because this checkpoint stores a full model object.
                    # Only do this because you downloaded the file yourself and trust its source.
                    ckpt = torch.load(
                        checkpoint_path,
                        map_location=self.device,
                        weights_only=False,
                    )

                    if isinstance(ckpt, nn.Module):
                        # Directly use the loaded model (original OnsetsAndFrames)
                        self.model = ckpt.to(self.device)
                        self.model.eval()
                        print("[INFO] Loaded full nn.Module checkpoint (OnsetsAndFrames).")
                    elif isinstance(ckpt, dict):
                        # Fallback: treat as state_dict or Lightning-style dict
                        state = ckpt.get("state_dict", ckpt)
                        self.model = OFOrig(OF_N_MELS, OF_N_PITCHES).to(self.device)
                        fixed = {k.replace("module.", ""): v for k, v in state.items()}
                        missing, unexpected = self.model.load_state_dict(fixed, strict=False)
                        print("[INFO] Loaded dict checkpoint with strict=False")
                        if missing:
                            print("[WARN] Missing keys:", missing[:10], "...")
                        if unexpected:
                            print("[WARN] Unexpected keys:", unexpected[:10], "...")
                    else:
                        print(f"[WARN] Unexpected checkpoint type: {type(ckpt)}, using random init.")

            except Exception as e:
                # Don't kill the server—log and continue with random weights
                import traceback
                print("[WARN] Failed to load checkpoint:", repr(e))
                traceback.print_exc()
                
        # ---- SAFE dynamic quantization (optional) ----
        # On your Mac, quantization engine is likely "none" / "NoQEngine",
        # so this will just log and keep the float32 model.
        import torch.nn as nn

        qengine = None
        try:
            if hasattr(torch.backends, "quantized"):
                qengine = torch.backends.quantized.engine
        except Exception:
            qengine = None

        if self.device.type == "cpu" and qengine and qengine not in ("none", "NoQEngine"):
            try:
                print(f"[INFO] Enabling dynamic quantization (engine={qengine})...")
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.GRU, nn.Linear},
                    dtype=torch.qint8,
                )
                print("[INFO] Dynamic quantization enabled successfully.")
            except Exception as e:
                print("[WARN] Dynamic quantization failed; using float32 instead:", repr(e))
        else:
            print(f"[INFO] Quantization engine={qengine!r}; skipping dynamic quantization (float32 model).")

    def _prep_audio(self, y: np.ndarray, sr: int) -> torch.Tensor:
        x = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,T]
        if sr != self.sr_model:
            x = torchaudio.functional.resample(x, sr, self.sr_model)
        return x

    def _audio_to_mel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use original Onsets & Frames mel front-end.

        x: [1, T] float32, assumed in [-1, 1]
        returns: [1, T_frames, OF_N_MELS]  (time-major)
        """
        if x.dim() != 2:
            x = x.view(1, -1)

        # O&F melspectrogram expects (B, T) and returns (B, n_mels, frames)
        mel = self.of_melspec(x)  # [B, n_mels, frames]

        # Transpose to [B, frames, n_mels] so it matches OnsetsAndFrames.forward
        if mel.dim() == 3:
            mel = mel.transpose(1, 2).contiguous()  # [B, T_frames, n_mels]

        return mel



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
        # 1) audio -> tensor
        x = self._prep_audio(y, sr)       # [1, T']
        # 2) tensor -> mel (your torchaudio-based frontend)
        mel = self._audio_to_mel(x)       # shape adapted for OFOrig

        # 3) run original Onsets & Frames model
        outputs = self.model(mel)

        # Original O&F variations:
        # - (onset, offset, frame, velocity)
        # - (onset, offset, frame, velocity, something_extra)
        # We only care about onset + frame.
        if isinstance(outputs, (list, tuple)):
            if len(outputs) >= 3:
                onset_logits = outputs[0]
                frame_logits = outputs[2]
            elif len(outputs) == 2:
                onset_logits, frame_logits = outputs
            else:
                raise RuntimeError(
                    f"Unexpected number of outputs from O&F model: {len(outputs)}"
                )
        else:
            raise RuntimeError(
                "Onsets & Frames model returned a non-tuple output; "
                "expected (onset, offset, frame, ...)."
            )

        return self._postprocess(onset_logits, frame_logits, sr=self.sr_model)



    @torch.no_grad()
    def transcribe_chunked(
        self,
        y: np.ndarray,
        sr: int,
        chunk_sec: float = 2.0,
        hop_sec: float = 1.0,
        onset_filt: int = 3,
        frame_filt: int = 5,
        th_on_hi: float = 0.5,
        th_on_lo: float = 0.3,
        th_fr: float = 0.5,
    ) -> list:
        """
        TEMP: for now, just fall back to full-mode transcription.

        This keeps your API shape identical and avoids crashes while
        we get the core model working. Later you can re-introduce a
        proper overlap-add chunked implementation.
        """
        return self.transcribe(y, sr)


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
