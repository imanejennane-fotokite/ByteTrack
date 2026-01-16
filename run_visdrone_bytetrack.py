import os
import cv2
import numpy as np
import sys
import torch

sys.path.append("/home/imane/object_tracking_project/tracking/trackers/yolox")

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.byte_tracker import STrack


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------

def xyxy_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if inter <= 0:
        return 0.0

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return inter / float(box1_area + box2_area - inter)


def load_xyxy_from_txt(path):
    with open(path, "r") as f:
        line = f.readline()
        delimiter = "," if "," in line else None

    arr = np.loadtxt(path, dtype=float, delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


# -------------------------------------------------------
# Main SOT Tracking Script (ByteTrack edition)
# -------------------------------------------------------

def main():

    # ---------------------------
    # Dataset Paths
    # ---------------------------
    DATA_ROOT = "/home/imane/object_tracking_project/datasets/VisDrone-SOT/VisDrone2019-SOT-val-new"
    IMG_ROOT = os.path.join(DATA_ROOT, "sequences")
    GT_ROOT = os.path.join(DATA_ROOT, "annotations_XYXY")
    DET_ROOT = "/home/imane/object_tracking_project/datasets_detector/yolo_progressive_pred/VisDrone2019-SOT-val_XYXY"

    OUT_ROOT = "/home/imane/object_tracking_project/tracking/trackers_results/ByteTrack/ByteTrack-VisDrone-SOT_XYXY"
    os.makedirs(OUT_ROOT, exist_ok=True)

    # ---------------------------
    # Build ByteTrack (SOT-friendly)
    # ---------------------------
    class BTArgs:
        track_thresh = 0.2
        track_buffer = 20
        match_thresh = 0.4
        mot20 = False

    bt_args = BTArgs()
    tracker = BYTETracker(bt_args, frame_rate=30)

    # ---------------------------
    # Iterate Over All Sequences
    # ---------------------------
    sequences = sorted(os.listdir(IMG_ROOT))

    for seq_name in sequences:
        print(f"\n=== Processing sequence: {seq_name} ===")

        out_path = os.path.join(OUT_ROOT, f"{seq_name}.txt")

        print(f"output path issss: {out_path}")
        # Skip if result already exists
        if os.path.isfile(out_path):
            print(f"Result already exists for {seq_name}. Skipping.")
            continue

        seq_img_dir = os.path.join(IMG_ROOT, seq_name)
        gt_path = os.path.join(GT_ROOT, f"{seq_name}.txt")
        det_dir = os.path.join(DET_ROOT, seq_name)

        if not os.path.isfile(gt_path):
            print(f"GT missing: {gt_path}")
            continue
        if not os.path.isdir(det_dir):
            print(f"YOLO detections missing: {det_dir}")
            continue

        # Load GT for frame 1
        gt_boxes = load_xyxy_from_txt(gt_path)
        init_gt = gt_boxes[0]

        # -----------------------------
        # Select YOLO detection for frame 1
        # -----------------------------
        det_first_path = os.path.join(det_dir, "img0000001.txt")

        if not os.path.isfile(det_first_path):
            print("Missing first-frame YOLO. Skip.")
            continue

        dets = load_xyxy_from_txt(det_first_path)[:, :4]
        best_det = None
        best_iou = -1
        for d in dets:
            iou = xyxy_iou(init_gt, d)
            if iou > best_iou:
                best_iou = iou
                best_det = d

        if best_det is None:
            print("No matching detection for frame 1.")
            continue

        print(f"Initial YOLO vs GT IoU = {best_iou:.4f}")

        # ---------------------------
        # SOT State
        # ---------------------------
        target_id = None    # ByteTrack ID we follow
        last_confirmed_xyxy = best_det.tolist()

        predictions = []

        # ---------------------------
        # Process frames
        # ---------------------------
        frame_list = sorted(f for f in os.listdir(seq_img_dir) if f.endswith('.jpg'))

        for frame_name in frame_list:

            # Load frame
            frame_path = os.path.join(seq_img_dir, frame_name)
            frame = cv2.imread(frame_path)
            if frame is None:
                predictions.append([0, 0, 0, 0])
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # Load YOLO detections
            det_path = os.path.join(det_dir, frame_name.replace(".jpg", ".txt"))
            bt_inputs = []
            if os.path.isfile(det_path):
                arr = load_xyxy_from_txt(det_path)
                if arr.size > 0:
                    for d in arr:
                        x1,y1,x2,y2,score = d[:5]
                        bt_inputs.append([x1,y1,x2,y2,score,0])

            bt_inputs = np.array(bt_inputs)

            # FIX: convert to torch tensor
            if bt_inputs.size > 0:
                bt_inputs = torch.from_numpy(bt_inputs).float()
            else:
                # still pass an empty tensor (important)
                bt_inputs = torch.zeros((0, 6)).float()

            # ---------------------------
            # ByteTrack Update
            # ---------------------------
            online_targets = tracker.update(
                bt_inputs,
                img_info=[h, w],
                img_size=[h, w]
            )

            if not hasattr(tracker, "frame_id"):
                tracker.frame_id = 1
            else:
                tracker.frame_id += 1

            # ---------------------------
            # Choose our single-object track (SOT)
            # ---------------------------
            if target_id is None:
                # Pick the ByteTrack track matching initial YOLO
                best_tid = None
                best_tiou = -1
                for t in online_targets:
                    iou = xyxy_iou(best_det, t.tlbr)
                    if iou > best_tiou:
                        best_tiou = iou
                        best_tid = t

                if best_tid is not None:
                    target_id = best_tid.track_id

            # ---------------------------
            # Get predicted box from ByteTrack (if tracked)
            # ---------------------------
            pred_xyxy = [0,0,0,0]

            for t in online_targets:
                if t.track_id == target_id:
                    x, y, w0, h0 = t.tlwh
                    pred_xyxy = [x, y, x + w0, y + h0]
                    break

            # -----------------------------------
            # 1) LOST?
            # -----------------------------------
            lost = (pred_xyxy == [0, 0, 0, 0])

            # -----------------------------------
            # 2) Compute best YOLO for recovery
            # -----------------------------------
            best_yolo_det = None
            best_yolo_iou = -1

            if os.path.isfile(det_path):
                arr = load_xyxy_from_txt(det_path)
                if arr.size > 0:
                    for d in arr[:, :4]:
                        iou = xyxy_iou(last_confirmed_xyxy, d)
                        if iou > best_yolo_iou:
                            best_yolo_iou = iou
                            best_yolo_det = d.tolist()

            # -----------------------------------
            # 3) LOST TRACK RECOVERY
            # -----------------------------------
            if lost:
                if best_yolo_det is not None:

                    pred_xyxy = best_yolo_det
                    last_confirmed_xyxy = best_yolo_det

                    x1, y1, x2, y2 = best_yolo_det
                    w = x2 - x1
                    h = y2 - y1

                    new_strack = STrack(
                        tlwh=np.array([x1, y1, w, h], dtype=float),
                        score=0.9
                    )
                    new_strack.activate(tracker.kalman_filter, frame_id=0)

                    tracker.tracked_stracks = [new_strack]
                    tracker.lost_stracks = []
                    tracker.removed_stracks = []

                    target_id = new_strack.track_id

                else:
                    pred_xyxy = last_confirmed_xyxy

            # -----------------------------------
            # 4) Drift correction
            # -----------------------------------
            IOU_YOLO_MATCH = 0.3
            IOU_PRED_MATCH = 0.2

            use_yolo = False

            if best_yolo_det is not None:
                yolo_ok = best_yolo_iou > IOU_YOLO_MATCH
                iou_pred_vs_yolo = xyxy_iou(pred_xyxy, best_yolo_det)

                if iou_pred_vs_yolo < IOU_PRED_MATCH:
                    use_yolo = True
                if not yolo_ok:
                    use_yolo = False

            if use_yolo and best_yolo_det is not None:
                pred_xyxy = best_yolo_det

            # -----------------------------------
            # 5) Save and update SOT state
            # -----------------------------------
            last_confirmed_xyxy = pred_xyxy
            predictions.append(pred_xyxy)

        # ---------------------------
        # Save results
        # ---------------------------
        out_path = os.path.join(OUT_ROOT, f"{seq_name}.txt")
        np.savetxt(out_path, np.array(predictions), fmt="%.2f")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
