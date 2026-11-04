"""Fetch chosen Charades 480p videos out of the 16 GB official zip using HTTP range requests (only the members' bytes are downloaded).
Usage: python ag_fetch.py VID.mp4 ...   -> ag_data/videos/VID.mp4"""
import io, os, sys, urllib.request, zipfile
URL = "https://ai2-public-datasets.s3.amazonaws.com/charades/Charades_v1_480.zip"
class Http(io.RawIOBase):
    def __init__(s): s.pos = 0; s.n = int(urllib.request.urlopen(urllib.request.Request(URL, method="HEAD")).headers["Content-Length"])
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s, o, w=0): s.pos = o if w == 0 else s.pos + o if w == 1 else s.n + o; return s.pos
    def read(s, k=-1):
        end = s.n if k < 0 else min(s.pos + k, s.n)
        if end <= s.pos: return b""
        d = urllib.request.urlopen(urllib.request.Request(URL, headers={"Range": f"bytes={s.pos}-{end - 1}"})).read(); s.pos += len(d); return d
    def readinto(s, b): d = s.read(len(b)); b[:len(d)] = d; return len(d)
if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ag_data", "videos"); os.makedirs(out, exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(Http(), 1 << 20)); names = {os.path.basename(i.filename): i for i in z.infolist()}
    for v in sys.argv[1:]:
        if v not in names: print("missing", v); continue
        with z.open(names[v]) as src, open(os.path.join(out, v), "wb") as dst: dst.write(src.read())
        print(v, names[v].file_size)
