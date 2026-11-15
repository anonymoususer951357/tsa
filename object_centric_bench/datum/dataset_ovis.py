from collections import defaultdict
from pathlib import Path
import json
import pickle as pkl
import time

from einops import rearrange
from pycocotools import mask as maskUtils
import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util_datum import draw_segmentation_np, mask_segment_to_bbox_np

class OVIS(ptud.Dataset):

    def __init__(
        self,
        data_file,
        extra_keys=["segment", "bbox", "clazz"],
        transform=lambda **_: _,
        base_dir: Path = None,
        ts=None,
    ):
        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self_keys = pkl.loads(txn.get(b"__keys__"))
        print(len(self_keys))

        if ts is None:
            self.keys = [[_, None] for _ in self_keys]
        else:
            self.keys = []
            print(f"[{__class__.__name__}] slicing samples in dataset...")
            t0 = time.time()
            for key in self_keys:
                with env.begin(write=False) as txn:
                    sample = pkl.loads(txn.get(key))
                t = len(sample["video"])
                if t < ts:
                    continue
                num = int(np.ceil(t / ts))
                for i in range(num):
                    start = (i * ts) if (i + 1 < num) else (t - ts)
                    end = start + ts
                    if end > t:
                        start = t - ts
                    self.keys.append([key, start])
            print(len(self.keys))
            print(f"[{__class__.__name__}] {time.time() - t0}")

        env.close()

        self.extra_keys = extra_keys
        self.transform = transform
        self.ts = ts

    def __getitem__(self, index):

        if not hasattr(self, "env"):
            self.env = lmdb_open_read(self.data_file)

        key, start = self.keys[index]
        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(key))
        sample1 = {}

        if self.ts is None:
            end = None
        else:
            end = start + self.ts

        video = sample0["video"][start:end]
        video = np.array(
            [
                cv2.cvtColor(
                    cv2.imdecode(np.frombuffer(_, "uint8"), cv2.IMREAD_UNCHANGED),
                    cv2.COLOR_BGR2RGB,
                )
                for _ in video
            ]
        )
        video = pt.from_numpy(video).permute(0, 3, 1, 2)
        sample1["video"] = video

        if "segment" in self.extra_keys:
            segment = sample0["segment"][start:end]
            segment = np.array([cv2.imdecode(_, cv2.IMREAD_GRAYSCALE) for _ in segment])
            segment = pt.from_numpy(segment)
            sample1["segment"] = segment

            if "clazz" in self.extra_keys:
                clazz = pt.from_numpy(sample0["clazz"][start:end])
                sample1["clazz"] = clazz

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]
            if "clazz" in self.extra_keys:
                s0 = clazz.shape[1] + 1

                segment3 = ptnf.one_hot(segment2.long(), s0).bool()
            else:
                segment3 = ptnf.one_hot(segment2.long()).bool()

            t, h, w, s = segment3.shape

            cond = segment3.any([0, 1, 2])

            segment3 = segment3[:, :, :, cond]
            sample2["segment"] = segment3

            if "bbox" in self.extra_keys:
                segment3_ = rearrange(
                    segment3[:, :, :, 1:], "t h w s -> h w (t s)"
                )
                bbox2_ = pt.from_numpy(
                    mask_segment_to_bbox_np(segment3_.numpy())
                ).float()
                bbox2 = rearrange(bbox2_, "(t s) c -> t s c", t=t)
                bbox2[:, :, 0::2] /= w
                bbox2[:, :, 1::2] /= h
                sample2["bbox"] = bbox2

            if "clazz" in self.extra_keys:
                if cond.shape[0] - 1 != sample2["clazz"].shape[1]:
                    print(cond.shape[0], sample2["clazz"].shape[1])
                    print()
                clazz2 = sample2["clazz"][:, cond[1:]]
                sample2["clazz"] = clazz2

        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def convert_dataset(
        src_dir,
        dst_dir=Path("ovis"),
        splits=None,
    ):

        dst_dir.mkdir(parents=True, exist_ok=True)

        if splits is None:
            splits = dict(
                train=("annotations_train_split.json", "train"),
                val=("annotations_val_split.json", "train"),
            )

        for split, (annot_fn, video_sub) in splits.items():
            annot_file = src_dir / annot_fn
            video_fold = src_dir / video_sub
            if not annot_file.exists():
                print(f"[{__class__.__name__}] skip split {split!r}: {annot_file} not found")
                continue
            if not video_fold.exists():
                print(f"[{__class__.__name__}] skip split {split!r}: {video_fold} not found")
                continue
            print(split, annot_fn)
            with open(annot_file, "r") as fi:
                annot = json.load(fi)

            video_infos = {}
            for vinfo in annot["videos"]:
                video_infos[vinfo["id"]] = vinfo

            has_gt = annot.get("annotations") is not None
            track_infos = defaultdict(list)
            if has_gt:
                for tinfo in annot["annotations"]:
                    track_infos[tinfo["video_id"]].append(tinfo)
            else:
                print(
                    f"[{__class__.__name__}] split {split!r} has no ground-truth"
                    f" annotations; writing video-only entries."
                )

            lmdb_file = dst_dir / f"{split}.lmdb"
            lmdb_env = lmdb_open_write(lmdb_file)

            keys = []
            txn = lmdb_env.begin(write=True)
            t0 = time.time()

            cnt = 0
            video_iter = (
                track_infos.items() if has_gt
                else [(vid, []) for vid in video_infos.keys()]
            )
            for vid, track_info in video_iter:
                if has_gt and len(track_info) == 0:
                    continue

                vinfo = video_infos[vid]
                frame_fns = vinfo["file_names"]
                video_b = [(video_fold / _).read_bytes() for _ in frame_fns]

                if has_gt:
                    t = len(track_info[0]["segmentations"])
                    h = track_info[0]["height"]
                    w = track_info[0]["width"]
                    s = len(track_info)

                    assert all(h == _["height"] for _ in track_info)
                    assert all(w == _["width"] for _ in track_info)
                    assert all(t == len(_["segmentations"]) for _ in track_info)
                    assert t == len(video_b)

                    assert s + 1 < 256, f"too many objects ({s}) in video {vid}"

                    segment = np.zeros([t, h, w], "uint8")

                    clazz = np.zeros([t, s], "uint8")
                    for j, track in enumerate(track_info):
                        mask = __class__.rle_to_mask(track, h, w)
                        assert set(np.unique(mask)) <= {0, 1}
                        mask = mask.astype("bool")
                        clz = track["category_id"]
                        assert clz > 0
                        segment[mask] = j + 1
                        clazz[:, j] = clz

                    assert np.unique(segment).max() > 0
                    assert np.unique(clazz).min() > 0
                else:
                    t = len(video_b)
                    h = vinfo["height"]
                    w = vinfo["width"]
                    s = 0
                    segment = np.zeros([t, h, w], "uint8")
                    clazz = np.zeros([t, s], "uint8")

                sample_key = f"{cnt:06d}".encode("ascii")
                keys.append(sample_key)

                sample_dict = dict(
                    video=video_b,
                    segment=[
                        cv2.imencode(".webp", _)[1] for _ in segment
                    ],
                    clazz=clazz,
                )
                txn.put(sample_key, pkl.dumps(sample_dict))

                if (cnt + 1) % 64 == 0:
                    print(f"{cnt + 1:06d}")
                    txn.commit()
                    txn = lmdb_env.begin(write=True)

                cnt += 1

            txn.commit()
            if cnt > 0:
                print((time.time() - t0) / cnt)

            txn = lmdb_env.begin(write=True)
            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            lmdb_env.close()

    @staticmethod
    def rle_to_mask(track, h, w):
        masks = []
        for frameId in range(len(track["segmentations"])):
            brle = track["segmentations"][frameId]
            if brle is None:
                mask = np.zeros([h, w], "uint8")
            else:
                if type(brle) == list:
                    rles = maskUtils.frPyObjects(brle, h, w)
                    rle = maskUtils.merge(rles)
                elif type(brle["counts"]) == list:
                    rle = maskUtils.frPyObjects(brle, h, w)
                else:
                    rle = brle
                mask = maskUtils.decode(rle)
            masks.append(mask)
        return np.array(masks)

    @staticmethod
    def visualiz(video, segment=None, bbox=None, clazz=None, wait=0):

        assert video.ndim == 4 and video.shape[3] == 3 and video.dtype == np.uint8
        t, h, w, c = video.shape

        segment_viz = None
        if segment is not None and segment.shape[3]:
            assert segment.ndim == 4 and segment.dtype == bool

            if bbox is not None:
                assert (
                    bbox.ndim == 3 and bbox.shape[2] == 4 and bbox.dtype == np.float32
                )
                if clazz is not None:
                    assert clazz.shape[:2] == bbox.shape[:2]

                bbox[:, :, 0::2] *= w
                bbox[:, :, 1::2] *= h
                bbox = bbox.astype("int")

            if clazz is not None:
                assert clazz.ndim == 2 and clazz.dtype == np.uint8
                if bbox is not None:
                    assert clazz.shape[:2] == bbox.shape[:2]

        c1 = (255, 255, 255)
        frames = []
        segments_viz = []

        for ti, frame in enumerate(video):
            cv2.imshow("v", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frames.append(frame)

            if segment is not None and segment.shape[3]:
                seg_i = segment[ti, :, :, :]
                seg_i_viz = draw_segmentation_np(frame, seg_i, alpha=0.75)

                if bbox is not None:
                    for box in bbox[ti, :, :]:
                        seg_i_viz = cv2.rectangle(seg_i_viz, box[:2], box[2:], color=c1)

                if clazz is not None:
                    for ci, clz in enumerate(clazz[ti, :]):
                        msk = seg_i[:, :, ci + 1]
                        total = float(np.sum(msk))
                        if total == 0:
                            continue
                        ys, xs = np.indices(msk.shape)
                        cx = int(round((xs * msk).sum() / total))
                        cy = int(round((ys * msk).sum() / total))

                        seg_i_viz = cv2.putText(
                            seg_i_viz,
                            f"{clz}",
                            [cx, cy],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            [255] * 3,
                        )

                cv2.imshow("s", cv2.cvtColor(seg_i_viz, cv2.COLOR_RGB2BGR))
                segments_viz.append(seg_i_viz)

            cv2.waitKey(wait)

        return frames, segments_viz
