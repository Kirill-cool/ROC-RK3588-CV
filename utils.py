import json
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np

import addons.storages as strgs
from addons.byte_tracker import BYTETracker, draw_info, tracking
from addons.pulse_counter import Monitor
from base import Rk3588

CONFIG_FILE = str(Path(__file__).parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


def fill_storages(
        rk3588: Rk3588,
        tracker: Union[BYTETracker, None],
        raw_img_strg: strgs.ImageStorage,
        inf_img_strg: strgs.ImageStorage,
        dets_strg: strgs.DetectionsStorage,
        start_time: float
):
    """Fill storages with raw frames, frames with bboxes, numpy arrays with
    detctions

    Args
    -----------------------------------
    rk3588: Rk3588
        Object of Rk3588 class for getting data after inference
    raw_img_strg: storages.ImageStorage
        Object of ImageStorage for storage raw frames
    inf_img_strg: storages.ImageStorage
        Object of ImageStorage for storage inferenced frames
    dets_strg: storages.DetectionsStorage
        Object of DetectionsStorage for numpy arrays with detctions
    start_time: float
        Program start time
    tracker: BYTETracker | None
        detections tracker
    -----------------------------------
    """
    while True:
        output = rk3588.get_data()
        if output is not None:
            raw_frame, inferenced_frame, detections, frame_id = output
            # Bytetracker
            if tracker is not None and detections is not None:
                detections = tracking(
                    bytetracker=tracker,
                    dets=detections,
                    frame_shape=inferenced_frame.shape[:2]
                )
                if detections is not None:
                    draw_info(
                        frame=inferenced_frame,
                        dets=detections
                    )
            raw_img_strg.set_data(
                data=raw_frame,
                id=frame_id,
                start_time=start_time
            )
            cv2.rectangle(
                img=inferenced_frame,
                pt1=(316, 50),
                pt2=(324, 58),
                color=(128, 0, 0),
                thickness=4
            )
            inf_img_strg.set_data(
                data=inferenced_frame,
                id=frame_id,
                start_time=start_time
            )
            dets_strg.set_data(
                data=detections, # type: ignore
                id=frame_id,
                start_time=start_time
            )


def fill_combined_storage(
        rk3588_list: list[Rk3588],
        combined_img_strg: strgs.Storage,
        start_time: float
):
    last_frames = [None for _ in rk3588_list]
    last_frame_ids = [-1 for _ in rk3588_list]
    last_draw_time = None
    fps = 0.0
    min_fps = None
    max_fps = None
    combined_id = 0
    while True:
        updated_any = False
        for idx, rk in enumerate(rk3588_list):
            output = rk.get_data()
            if output is not None:
                _raw_frame, inferenced_frame, _detections, frame_id = output
                if frame_id != last_frame_ids[idx]:
                    last_frames[idx] = inferenced_frame
                    last_frame_ids[idx] = frame_id
                    updated_any = True
        if not updated_any:
            continue
        if all(frame is None for frame in last_frames):
            continue
        base_frame = next(
            (frame for frame in last_frames if frame is not None),
            None
        )
        if base_frame is None:
            continue
        for i, frame in enumerate(last_frames):
            if frame is None:
                blank = np.zeros_like(base_frame)
                cv2.putText(
                    img=blank,
                    text="NO SIGNAL",
                    org=(
                        int(blank.shape[1] / 2 - 100),
                        int(blank.shape[0] / 2)
                    ),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                last_frames[i] = blank
        heights = [frame.shape[0] for frame in last_frames]  # type: ignore
        target_height = min(heights)
        resized_frames = []
        for frame in last_frames:  # type: ignore
            if frame.shape[0] != target_height:
                new_width = int(frame.shape[1] * target_height / frame.shape[0])
                frame = cv2.resize(frame, (new_width, target_height))
            resized_frames.append(frame)
        combined = np.hstack(resized_frames)
        now = time.time()
        if last_draw_time is not None:
            delta = now - last_draw_time
            if delta > 0:
                fps = 1.0 / delta
                if min_fps is None or fps < min_fps:
                    min_fps = fps
                if max_fps is None or fps > max_fps:
                    max_fps = fps
        last_draw_time = now
        cv2.putText(
            img=combined,
            text=f"fps: {fps:.2f}",
            org=(5, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        if min_fps is not None:
            cv2.putText(
                img=combined,
                text=f"min_fps: {min_fps:.2f}",
                org=(5, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        if max_fps is not None:
            cv2.putText(
                img=combined,
                text=f"max_fps: {max_fps:.2f}",
                org=(5, 75),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        target_h, target_w = combined_img_strg._storage.shape[1:3]
        if combined.shape[0] != target_h or combined.shape[1] != target_w:
            combined = cv2.resize(combined, (target_w, target_h))
        combined_img_strg.set_data(
            data=combined,
            id=combined_id,
            start_time=start_time
        )
        combined_id += 1


def do_counting(
        inf_img_strg: strgs.ImageStorage,
        dets_strg: strgs.DetectionsStorage,
        counters_strg: strgs.Storage,
        pulse_monitor: Monitor
):
    stored_data_amount = cfg["storages"]["stored_data_amount"]
    while True:
        last_index = dets_strg.get_last_index()
        dets = dets_strg.get_data_by_index(last_index % stored_data_amount)
        if dets is not None:
            pulse_monitor.update(dets)
        img = inf_img_strg.get_data_by_index(last_index % stored_data_amount)
        if pulse_monitor.signal:
            cv2.rectangle(
                img=img,
                pt1=(316, 50),
                pt2=(324, 58),
                color=(0, 0, 128),
                thickness=8
            )
            counters_strg.set_data(
                data=pulse_monitor.up_counter,
                id=0
            )


def show_frames_localy(
        inf_img_strg: strgs.ImageStorage,
        start_time: float
):
    """Show inferenced frames with fps on device
    
    Args
    -----------------------------------
    inf_img_strg: storages.ImageStorage
        Object of ImageStorage for storage inferenced frames
    start_time: float
        Program start time
    -----------------------------------
    """
    cur_index = -1
    counter = 0
    calculated = False
    begin_time = time.time()
    fps = 0
    stored_data_amount = cfg["storages"]["stored_data_amount"]
    while True:
        last_index = inf_img_strg.get_last_index()
        if cfg["debug"]["showed_frame_id"] and cur_index != last_index:
            with open(cfg["debug"]["showed_id_file"], 'a') as f:
                f.write(
                    "{}\t{:.3f}\n".format(
                        cur_index,
                        time.time() - start_time
                    )
                )
        print(
            "cur - {} last - {}".format(
                cur_index,
                last_index
            ),
            end='\r'
        )
        frame = inf_img_strg.get_data_by_index(last_index % stored_data_amount)
        if cfg["camera"]["show"]:
            cv2.putText(
                img=frame,
                text="{:.2f}".format(fps),
                org=(5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        if last_index > cur_index:
            counter += 1
            cur_index = last_index
        if counter % 60 == 0 and not calculated:
            calculated = True
            fps = 60/(time.time() - begin_time)
            begin_time = time.time()
        if counter % 60 != 0:
            calculated = False
