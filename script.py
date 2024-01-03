import argparse
from typing import Dict, List, Set, Tuple, Optional  


import cv2
import numpy as np
import time

from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.default()

ZONE_IN_POLYGONS = [
    np.array([[260, 611], [308, 570], [318, 585], [270, 627]]), #bottom left
    np.array([[270, 631], [328, 583], [344, 601], [291, 648]]),
    np.array([[297, 653], [354, 600], [364, 624], [311, 672]]),
    np.array([[330, 694], [392, 642], [375, 625], [314, 678]]),
    np.array([[513, 4], [562, 40], [543, 67], [491, 26]]), #top left
    np.array([[462, 9], [506, 46], [474, 93], [414, 41]]),
    np.array([[383, 22], [440, 71], [422, 90], [370, 44]]),
    np.array([[366, 48], [344, 69], [397, 109], [415, 92]]),
    np.array([[1094, 149], [1114, 167], [1064, 217], [1044, 205]]), #top right
    np.array([[1095, 149], [1047, 106], [1008, 150], [1058, 183]]),
    np.array([[1020, 88], [1044, 102], [1012, 140], [991, 126]]), 
    np.array([[1023, 78], [1003, 63], [957, 111], [976, 133]]),
    np.array([[779, 691], [798, 672], [828, 695], [803, 715]]), #bottom right
    np.array([[804, 668], [839, 618], [893, 664], [860, 713]]),
    np.array([[895, 662], [919, 645], [973, 692], [957, 715]]),
    np.array([[1003, 713], [1018, 688], [967, 643], [957, 672]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[274, 221], [355, 129], [395, 166], [318, 261]]),
    np.array([[504, 711], [434, 641], [489, 594], [558, 668]]),
    np.array([[1012, 460], [1077, 517], [1001, 618], [942, 566]]),
    np.array([[754, 76], [801, 25], [905, 93], [848, 153]]),
]

ZONE_IN_SCOOT = [
    np.array([[421, 502], [479,561], [522, 529], [457, 462]]),
    np.array([[791, 666], [877, 568], [832, 530], [755, 630]]),
    np.array([[931, 321], [967, 283], [874, 201], [839, 239]]),
    np.array([[547, 241], [509, 205], [590, 113], [631, 145]]),
]


class DetectionsManager:
    def __init__(self, correct_table: Dict[int, Set[int]]) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}
        self.correct_table = correct_table
        self.incorrect_counts: Dict[int, int] = {}  # Counter for incorrect IDs by lane
        self.incorrect_ids: Dict[int, List[int]] = {}  # List of incorrect IDs by lane
        
        # Dictionary to store entry times for each car in ZONE_IN_SCOOT
        self.entry_times: Dict[int, Dict[int, float]] = {}
        
        # Dictionary to store cars that have stayed more than 10 seconds in ZONE_IN_SCOOT
        self.stayed_too_much: Dict[int, List[int]] = {}  # List of incorrect IDs by lane


    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
        detections_in_scoot_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
                    
                     # Check correctness and update incorrect counts
                    if zone_out_id in self.correct_table and zone_in_id not in self.correct_table[zone_out_id]:
                        self.incorrect_counts.setdefault(zone_out_id, 0)
                        self.incorrect_counts[zone_out_id] += 1
                        
                        self.incorrect_ids.setdefault(zone_out_id, [])
                        self.incorrect_ids[zone_out_id].append(tracker_id)

        for zone_in_scoot_id, detections_in_scoot_zone in enumerate(detections_in_scoot_zones):
            for tracker_id, entry_time in self.get_entry_times(detections_in_scoot_zone):
                if entry_time is None:
                    # Car just entered the zone, record entry time
                    self.record_entry_time(zone_in_scoot_id, tracker_id)
                else:
                    # Car is already in the zone, check if it has stayed for more than 3 seconds
                    elapsed_time = time.time() - entry_time
                    if elapsed_time > 3:
                        # Car has stayed more than 3 seconds, record it
                        self.record_stayed_time(zone_in_scoot_id, tracker_id)


        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]
        
    def get_entry_times(self, detections: sv.Detections) -> List[Tuple[int, Optional[float]]]:
        entry_times = []
        for tracker_id in detections.tracker_id:
            zone_id = self.tracker_id_to_zone_id.get(tracker_id, -1)
            if zone_id in self.entry_times and tracker_id in self.entry_times[zone_id]:
                entry_times.append((tracker_id, self.entry_times[zone_id][tracker_id]))
            else:
                entry_times.append((tracker_id, None))
        return entry_times

    def record_entry_time(self, zone_id: int, tracker_id: int) -> None:
        if zone_id not in self.entry_times:
            self.entry_times[zone_id] = {}
        self.entry_times[zone_id][tracker_id] = time.time()

        
    def get_incorrect_ids(self, lane_id: int) -> List[int]:
        """Get the list of incorrect IDs for a specific lane."""
        return self.incorrect_ids.get(lane_id, [])
        
    def record_stayed_time(self, zone_id: int, tracker_id: int) -> None:
        # Record the car ID and the zone_in_scoot_id where it stayed too much
        self.stayed_too_much.setdefault(zone_id, [])
        self.stayed_too_much[zone_id].append(tracker_id)

    def get_stayed_too_much(self) -> Dict[int, List[int]]:
        """Get the list of cars that have stayed too much in a specific zone."""
        return self.stayed_too_much

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )
        
        self.zones_in_scoot = initiate_pSolygon_zones(
        ZONE_IN_SCOOT, self.video_info.resolution_wh, sv.Position.CENTER
        )
        
        correct_table = {
        	1: {1, 11, 12, 14, 15, 6},
        	2: {13, 10, 11, 7, 8},
        	3: {9, 7, 6, 4, 3},
        	4: {2, 3, 5, 15, 16},
        }

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager(correct_table)

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        
        correct_table = {
        	1: {1, 11, 12, 14, 15, 6},
        	2: {13, 10, 11, 7, 8},
        	3: {9, 7, 6, 4, 3},
        	4: {2, 3, 5, 15, 16},
        # Add more entries as needed
        }
        
        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break


            cv2.destroyAllWindows()
            
        # Print the IDs of cars that stayed too much per zone after processing all frames
        for zone_id, ids_set in self.detections_manager.get_stayed_too_much().items():
            print(f"Cars that stayed too much in Zone {zone_id}:")
            for car_id in ids_set:
                print(f"  Car ID: {car_id}")



            
        # After processing all frames
        for lane_id in correct_table.keys():
            incorrect_ids_for_lane = self.detections_manager.get_incorrect_ids(lane_id)
            print(f"Incorrect IDs for Lane {lane_id}: {incorrect_ids_for_lane}")


    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        
        # Iterate over the longer list of zones
        max_num_zones = max(len(self.zones_in), len(self.zones_out), len(self.zones_in_scoot))

        for i in range(max_num_zones):
            if i < len(self.zones_in):
                zone_in = self.zones_in[i]
                annotated_frame = sv.draw_polygon(annotated_frame, zone_in.polygon, COLORS.colors[i])
                zone_in_center = sv.get_polygon_center(polygon=zone_in.polygon)
                annotated_frame = sv.draw_text(
                    scene=annotated_frame,
                    text=str(i + 1),  # Polygon number (1-indexed)
                    text_anchor=zone_in_center,
                    background_color=COLORS.colors[i],
                )
        
            if i < len(self.zones_out):
                zone_out = self.zones_out[i]
                annotated_frame = sv.draw_polygon(annotated_frame, zone_out.polygon, COLORS.colors[i])
                zone_out_center = sv.get_polygon_center(polygon=zone_out.polygon)
                annotated_frame = sv.draw_text(
                    scene=annotated_frame,
                    text=str(i + 1),  # Polygon number (1-indexed)
                    text_anchor=zone_out_center,
                    background_color=COLORS.colors[i],
                )

            if i < len(self.zones_in_scoot):
                zone_in_scoot = self.zones_in_scoot[i]
                annotated_frame = sv.draw_polygon(annotated_frame, zone_in_scoot.polygon, COLORS.colors[i])
                zone_in_scoot_center = sv.get_polygon_center(polygon=zone_in_scoot.polygon)
                annotated_frame = sv.draw_text(
                    scene=annotated_frame,
                    text=str(i + 1),  # Polygon number (1-indexed)
                    text_anchor=zone_in_scoot_center,
                    background_color=COLORS.colors[i],
                )
                
                
        # Display the number of cars that stayed too much in each zone
        for zone_id, tracker_ids in self.detections_manager.get_stayed_too_much().items():
            zone_index = zone_id - 1  # Adjust index as zone IDs are 1-indexed
            zone_in_scoot = self.zones_in_scoot[zone_index]
        
            annotated_frame = sv.draw_polygon(annotated_frame, zone_in_scoot.polygon, COLORS.colors[zone_index])
            zone_in_scoot_center = sv.get_polygon_center(polygon=zone_in_scoot.polygon)

           # annotated_frame = sv.draw_text(
            #    scene=annotated_frame,
           #     text=str(zone_id),  # Display the actual zone ID
           #     text_anchor=zone_in_scoot_center,
           #     background_color=COLORS.colors[zone_index],
           # )

            # Display the number of cars that stayed too much in the zone
            stayed_too_much_count = len(tracker_ids)

            text_anchor_stayed = sv.Point(x=zone_in_scoot_center.x, y=zone_in_scoot_center.y + 40)

            annotated_frame = sv.draw_text(
                scene=annotated_frame,
                text=f"Cars stayed too much: {stayed_too_much_count}",
                text_anchor=text_anchor_stayed,
                background_color=COLORS.colors[zone_index],
            )        

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )
                    

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []
        detections_in_scoot_zones = []  

        max_num_zones = max(len(self.zones_in), len(self.zones_out), len(self.zones_in_scoot))

        for i in range(max_num_zones):
            if i < len(self.zones_in):
                zone_in = self.zones_in[i]
                detections_in_zone = detections[zone_in.trigger(detections=detections)]
                detections_in_zones.append(detections_in_zone)

            if i < len(self.zones_out):
                zone_out = self.zones_out[i]
                detections_out_zone = detections[zone_out.trigger(detections=detections)]
                detections_out_zones.append(detections_out_zone)

            if i < len(self.zones_in_scoot):
                zone_in_scoot = self.zones_in_scoot[i]
                detections_in_scoot_zone = detections[zone_in_scoot.trigger(detections=detections)]
                detections_in_scoot_zones.append(detections_in_scoot_zone)  

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones, detections_in_scoot_zones  # Update this line
        )

        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTrack"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()
