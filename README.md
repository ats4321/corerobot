# Core Robot

Learns a tabletop task from one demonstration video and performs it with a simulated Franka Panda arm in PyBullet, using Meta's SAM 3.1 and Muse Spark models.

See [PROJECT.md](PROJECT.md) for the full brief.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MODEL_API_KEY=...
```
