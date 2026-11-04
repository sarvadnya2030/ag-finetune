"""Action Genome -> VLM SFT data. One sample = one annotated frame image -> JSON of the visible objects' relations.
  python prepare_data.py --ann object_bbox_and_relationship.pkl --videos videos --split train --out data/train.jsonl [--max-videos 500 --per-video 8]
  python prepare_data.py --ann ... --list 500     # print 500 train video ids (fetch them with ag_fetch.py)"""
import argparse, collections, json, os, pickle, random, cv2
class U(pickle.Unpickler):  # restricted: official AG pkl only needs numpy/collections
    def find_class(s, m, n):
        if m.split(".")[0] in ("numpy", "collections", "builtins"): return super().find_class(m, n)
        raise pickle.UnpicklingError(f"blocked {m}.{n}")
PROMPT = ("List every visible object that the person is related to in this frame. Output JSON only: "
          '[{"object":<name>,"attention":[..],"spatial":[..],"contact":[..]}]')
def answer(objs):
    return json.dumps([dict(object=o["class"], attention=o["attention_relationship"] or [], spatial=o["spatial_relationship"] or [],
                            contact=o["contacting_relationship"] or []) for o in objs if o["visible"]])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ann", required=True); ap.add_argument("--videos", default="videos")
    ap.add_argument("--split", default="train"); ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--max-videos", type=int, default=0); ap.add_argument("--per-video", type=int, default=8)
    ap.add_argument("--short", type=int, default=448); ap.add_argument("--list", type=int, default=0); a = ap.parse_args()
    O = U(open(a.ann, "rb"), encoding="latin1").load(); by = collections.defaultdict(list)
    for k, objs in O.items():
        if objs and objs[0]["metadata"]["set"] == a.split and any(o["visible"] for o in objs): by[k.split("/")[0]].append(k)
    vids = sorted(by); random.Random(0).shuffle(vids)
    if a.list: print("\n".join(vids[:a.list])); raise SystemExit
    have = {v for v in vids if os.path.exists(os.path.join(a.videos, v))}; vids = [v for v in vids if v in have][:a.max_videos or None]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); img_dir = os.path.join(os.path.dirname(a.out) or ".", "img"); os.makedirs(img_dir, exist_ok=True)
    n = 0
    with open(a.out, "w") as f:
        for v in vids:
            ks = sorted(by[v]); ks = [ks[i] for i in sorted(random.Random(v).sample(range(len(ks)), min(a.per_video, len(ks))))]
            cap = cv2.VideoCapture(os.path.join(a.videos, v))
            for k in ks:
                cap.set(1, int(k.split("/")[1][:-4])); ok, im = cap.read()
                if not ok: continue
                s = a.short / min(im.shape[:2]); im = cv2.resize(im, None, fx=s, fy=s) if s < 1 else im
                p = os.path.join(img_dir, k.replace("/", "_").replace(".png", ".jpg")); cv2.imwrite(p, im)
                f.write(json.dumps(dict(id=k, image=os.path.abspath(p), prompt=PROMPT, answer=answer(O[k]))) + "\n"); n += 1
    print(f"{n} samples from {len(vids)} videos -> {a.out}")
