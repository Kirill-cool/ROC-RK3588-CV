import json
import time
from multiprocessing import Process
from pathlib import Path
from threading import Thread

import cv2
import numpy as np

import addons.storages as strgs
from addons.byte_tracker import BTArgs, BYTETracker
from addons.pulse_counter import Monitor
from addons.webui import WebUI
from addons.webui.utils import obj_imgs_to_str
from base import Rk3588
from utils import do_counting, fill_storages, show_frames_localy

CONFIG_FILE = str(Path(__file__).parent.absolute()) + "/config.json"
with open(CONFIG_FILE, 'r') as config_file:
    cfg = json.load(config_file)


def main():
    """Runs inference and addons (if mentions)
    Creating storages and sending data to them
    """
    start_time = time.time()
    sources = cfg["camera"]["source"]
    if isinstance(sources, list):
        rk3588 = [Rk3588(source=src) for src in sources]
        for rk in rk3588:
            rk.start()
    else:
        rk3588 = Rk3588()
        rk3588.start()
    if not cfg["storages"]["state"]:
        if isinstance(rk3588, list):
            video_writer = None
            video_file = None
            last_frames = [None for _ in rk3588]
            last_frame_ids = [-1 for _ in rk3588]
            last_draw_time = None
            fps = 0.0
            min_fps = None
            max_fps = None
            try:
                while True:
                    updated_any = False
                    for idx, rk in enumerate(rk3588):
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
                    cv2.imshow("frame", combined)
                    cv2.waitKey(1)

                    if video_writer is None:
                        videos_dir = Path.cwd() / "videos"
                        videos_dir.mkdir(exist_ok=True)
                        model_name = Path(cfg["inference"]["new_model"]).stem
                        video_file = str(
                            videos_dir / f"output_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                        )
                        height, width = combined.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(
                            video_file, fourcc, cfg["camera"]["fps"], (width, height)
                        )
                        print(f"[INFO] Video recording started: {video_file}")
                    video_writer.write(combined)
            except Exception as e:
                print("Main exception: {}".format(e))
                exit()
            finally:
                if video_writer is not None:
                    video_writer.release()
                    print(f"[INFO] Video saved: {video_file}")
                exit()
        try:
            while True:
                rk3588.show(start_time)
        except Exception as e:
            print("Main exception: {}".format(e))
            exit()
    raw_frames_storage = strgs.ImageStorage(
        "raw frames"
    )
    inferenced_frames_storage = strgs.ImageStorage(
        "inferenced frames"
    )
    detections_storage = strgs.DetectionsStorage()
    counters_storage = strgs.Storage(
        storage_name="counters",
        data_size=(1,),
        data_amount=len(cfg["inference"]["classes"]),
        data_type=int
    )
    tracker = None
    if cfg["bytetrack"]["state"]:
        bytetrack_args = BTArgs()
        tracker = BYTETracker(
            args=bytetrack_args,
            frame_rate=cfg["bytetrack"]["fps"]
        )
    fill_thread = Thread(
        target=fill_storages,
        kwargs={
            "rk3588": rk3588,
            "tracker": tracker,
            "raw_img_strg": raw_frames_storage,
            "inf_img_strg": inferenced_frames_storage,
            "dets_strg": detections_storage,
            "start_time": start_time
        },
        daemon=True
    )
    fill_thread.start()
    if cfg["pulse_counter"]["state"]:
        pulse_monitor = Monitor()
        counting_thread = Thread(
            target=do_counting,
            kwargs={
                "inf_img_strg": inferenced_frames_storage,
                "dets_strg": detections_storage,
                "counters_strg": counters_storage,
                "pulse_monitor": pulse_monitor
            },
            daemon=True
        )
        counting_thread.start()
    if cfg["webui"]["state"]:
        ui = WebUI(
            raw_img_strg=raw_frames_storage,
            inf_img_strg=inferenced_frames_storage,
            dets_strg=detections_storage,
            counters_strg=counters_storage,
            camera=rk3588._cam
        )
        try:
            obj_imgs_to_str()
            ui.start()
        except Exception as e:
            print("WebUI exception: {}".format(e))
        finally:
            fill_thread.join()
            if cfg["pulse_counter"]["state"]:
                counting_thread.join() # type: ignore
            counters_storage.clear_buffer()
            raw_frames_storage.clear_buffer()
            inferenced_frames_storage.clear_buffer()
            detections_storage.clear_buffer()
            exit()
    try:        
        show_frames_localy(
            inf_img_strg=inferenced_frames_storage,
            start_time=start_time
        )
    except Exception as e:
        print("Main exception: {}".format(e))
    finally:
        fill_thread.join()
        if cfg["pulse_counter"]["state"]:
            counting_thread.join() # type: ignore
        counters_storage.clear_buffer()
        raw_frames_storage.clear_buffer()
        inferenced_frames_storage.clear_buffer()
        detections_storage.clear_buffer()
        exit()


if __name__ == "__main__":
    main()
