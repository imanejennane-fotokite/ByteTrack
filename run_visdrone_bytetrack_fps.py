import os
import cv2
import numpy as np
import sys
import torch
import time # Importation de la librairie time pour le calcul des FPS
from ultralytics import YOLO
from turbojpeg import TurboJPEG

sys.path.append("/home/imane/object_tracking_project/tracking/trackers/yolox")

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.byte_tracker import STrack


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------
jpeg = TurboJPEG("/usr/lib/aarch64-linux-gnu/libturbojpeg.so")

def fast_read_rgb(path):
    """Lecture rapide de l'image et conversion BGR -> RGB"""
    with open(path, 'rb') as f:
        img = jpeg.decode(f.read())
    return img[..., ::-1]

def xyxy_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / float(box1_area + box2_area - inter)

# Fonction restaurée du script ByteTrack original
def load_xyxy_from_txt(path):
    with open(path, "r") as f:
        line = f.readline()
        delimiter = "," if "," in line else None

    # Utilisation de np.float32 pour uniformité avec le reste
    arr = np.loadtxt(path, dtype=np.float32, delimiter=delimiter)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def yolo_to_xyxy_fast(det):
    """Conversion des résultats YOLO Ultralytics en boxes et scores numpy"""
    if det.boxes.data is None or len(det.boxes.data) == 0:
        return np.empty((0,4), dtype=np.float32), np.empty(0, dtype=np.float32)
    arr = det.boxes.data.cpu().numpy().astype(np.float32)
    return arr[:, :4], arr[:, 4]


# -------------------------------------------------------
# Main SOT Tracking Script (ByteTrack with Online YOLO & FPS)
# -------------------------------------------------------

def main():

    # ---------------------------
    # Dataset Paths
    # ---------------------------
    DATA_ROOT = "/home/imane/object_tracking_project/datasets/VisDrone-SOT/VisDrone2019-SOT-val-new"
    IMG_ROOT = os.path.join(DATA_ROOT, "sequences")
    GT_ROOT = os.path.join(DATA_ROOT, "annotations_XYXY")

    OUT_ROOT = "/home/imane/object_tracking_project/tracking/trackers_results/ByteTrack/ByteTrack-FPS3"
    os.makedirs(OUT_ROOT, exist_ok=True)
    
    # ... (YOLO Model Loading) ...
    YOLO_PATH = "/home/imane/object_tracking_project/detection/models/yolo/Progressive/best_model_progressive/best.engine"
    print("\n🔹 Loading YOLO TensorRT…")
    yolo = YOLO(YOLO_PATH, task='detect')
    
    # Warmup YOLO
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = yolo(dummy, verbose=False, device=0)
    torch.cuda.synchronize()

    # ... (ByteTrack Setup) ...
    class BTArgs:
        track_thresh = 0.2
        track_buffer = 20
        match_thresh = 0.4
        mot20 = False

    bt_args = BTArgs()
    tracker = BYTETracker(bt_args, frame_rate=30)
    
    total_time = 0.0
    total_frames = 0

    # ---------------------------
    # Iterate Over All Sequences
    # ---------------------------
    sequences = sorted(os.listdir(IMG_ROOT))

    for seq_name in sequences:
        print(f"\n=== Processing sequence: {seq_name} ===")

        out_path = os.path.join(OUT_ROOT, f"{seq_name}.txt")

        if os.path.isfile(out_path):
            print(f"Result already exists for {seq_name}. Skipping.")
            continue

        seq_img_dir = os.path.join(IMG_ROOT, seq_name)
        gt_path = os.path.join(GT_ROOT, f"{seq_name}.txt")
        
        if not os.path.isfile(gt_path):
            print(f"GT missing: {gt_path}")
            continue
        
        frame_list = sorted(f for f in os.listdir(seq_img_dir) if f.endswith('.jpg'))
        if not frame_list:
            print(f"No images found for {seq_name}. Skipping.")
            continue
            
        num_frames = len(frame_list)
            
        # -----------------------------
        # Load GT for frame 1 (CORRIGÉ ET ROBUSTE)
        # -----------------------------
        gt_boxes = None 
        try:
            gt_boxes = load_xyxy_from_txt(gt_path) 
            
            if gt_boxes is None or gt_boxes.size == 0:
                print(f"Error loading GT for {seq_name}: File is empty or data is invalid.")
                continue

        except Exception as e:
             # Gère les erreurs de lecture de fichier
             print(f"Error loading GT for {seq_name}. Details: {e}. Skipping.")
             continue

        init_gt = gt_boxes[0]

        # -----------------------------
        # Select YOLO detection for frame 1 (Online)
        # -----------------------------
        frame1_path = os.path.join(seq_img_dir, frame_list[0])
        frame1 = fast_read_rgb(frame1_path)
        
        # ... (le reste du code est conservé) ...
        det1 = yolo(frame1, verbose=False, stream=False, device=0)[0]
        dets_xyxy, _ = yolo_to_xyxy_fast(det1)
        
        best_det = None
        best_iou = -1
        for d in dets_xyxy:
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
        target_id = None
        last_confirmed_xyxy = best_det.tolist()

        predictions = []

        # ---------------------------
        # Process frames (Timing starts here)
        # ---------------------------
        seq_start_time = time.time()

        for frame_name in frame_list:

            # Load frame (use fast_read_rgb)
            frame_path = os.path.join(seq_img_dir, frame_name)
            frame_rgb = fast_read_rgb(frame_path)
            
            if frame_rgb is None:
                predictions.append([0, 0, 0, 0])
                continue
            
            h, w = frame_rgb.shape[:2]

            # ---------------------------
            # Online YOLO Detection
            # ---------------------------
            bt_inputs = []
            
            # Appel YOLO en ligne
            det = yolo(frame_rgb, verbose=False, stream=False, device=0)[0]
            current_dets_xyxy, current_scores = yolo_to_xyxy_fast(det)
            
            # Formatage pour ByteTrack
            if current_dets_xyxy.size > 0:
                for d_xyxy, score in zip(current_dets_xyxy, current_scores):
                    bt_inputs.append([d_xyxy[0], d_xyxy[1], d_xyxy[2], d_xyxy[3], score, 0])

            bt_inputs = np.array(bt_inputs)
            best_yolo_det = None
            
            # FIX: convert to torch tensor
            if bt_inputs.size > 0:
                bt_inputs = torch.from_numpy(bt_inputs).float()
            else:
                bt_inputs = torch.zeros((0, 6)).float()

            # ---------------------------
            # ByteTrack Update
            # ---------------------------
            online_targets = tracker.update(
                bt_inputs,
                img_info=[h, w],
                img_size=[h, w]
            )

            # Frame ID management
            if not hasattr(tracker, "frame_id"):
                tracker.frame_id = 1
            else:
                tracker.frame_id += 1

            # ---------------------------
            # Choose our single-object track (SOT)
            # ---------------------------
            if target_id is None:
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
            # Get predicted box from ByteTrack
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

            if current_dets_xyxy.size > 0:
                for d in current_dets_xyxy:
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
        # Sequence FPS Calculation
        # ---------------------------
        seq_elapsed = time.time() - seq_start_time
        if seq_elapsed > 0:
            seq_fps = num_frames / seq_elapsed
            print(f"🔥 FPS for {seq_name}: {seq_fps:.2f} ({num_frames} frames in {seq_elapsed:.2f}s)")
        else:
            print(f"Sequence {seq_name} finished too quickly to calculate FPS.")
            
        total_time += seq_elapsed
        total_frames += num_frames
        
        # Clear CUDA cache between sequences
        torch.cuda.empty_cache()

        # ---------------------------
        # Save results
        # ---------------------------
        np.savetxt(out_path, np.array(predictions), fmt="%.2f")
        print(f"Saved: {out_path}")

    # ---------------------------------------------
    # GLOBAL FPS
    # ---------------------------------------------
    if total_frames > 0 and total_time > 0:
        global_fps = total_frames / total_time
        print("\n========================================")
        print(f"🚀 GLOBAL AVERAGE FPS: {global_fps:.2f}")
        print(f"   Total frames: {total_frames}")
        print(f"   Total time: {total_time:.2f}s")
        print("========================================\n")
    elif total_frames > 0:
        print("\n========================================")
        print(f"🚀 GLOBAL FPS: Time elapsed is zero or too small.")
        print("========================================\n")


if __name__ == "__main__":
    main()