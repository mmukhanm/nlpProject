"""Скачать неизменённый Amazon Cells TSV и проверить SHA-256."""
from pathlib import Path
import hashlib
import urllib.request

URL = "https://raw.githubusercontent.com/ctc316/uci-sentiment_labelled_sentences/b8411a53bfaf71dffbcbf3d5e1549d5c0c60ed3a/data/amazon_cells_labelled.txt"
SHA256 = "47003fc0a0d4840b00e96e715b6189bad09e7443a3da41c4cbe12ffc79f86ae3"
path = Path(__file__).resolve().parents[1] / "data/raw/amazon_cells_labelled.txt"
if path.exists():
    content = path.read_bytes()
else:
    with urllib.request.urlopen(URL, timeout=60) as response: content = response.read()
if hashlib.sha256(content).hexdigest() != SHA256:
    raise ValueError("Контрольная сумма не совпала; файл не сохранён")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(content)
print(f"Verified: {path} ({len(content)} bytes)")
