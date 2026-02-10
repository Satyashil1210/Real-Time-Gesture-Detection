# Dataset Format (Planned)

Base folders:
- `data/raw/` – raw captured frames (per-label subfolders).
- `data/processed/` – precomputed landmark sequences or feature vectors.

Example structure:

```text
data/
  raw/
    ok/
      ok_*.jpg
    rock/
      rock_*.jpg
    stop/
      stop_*.jpg
  processed/
    landmarks_ok.npz
    landmarks_rock.npz
