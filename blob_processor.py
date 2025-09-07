import numpy as np
import cv2
from collections import deque
import tempfile
import os

class BlobTracker:
    def __init__(self):
        self.avg_y_history = []
        self.prev_centers = None
    
    def interpolate_catmull_rom(self, points, resolution=12):
        def catmull_rom_spline(p0, p1, p2, p3, t):
            t2, t3 = t * t, t * t * t
            return 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2*p0 - 5*p1 + 4*p2 - p3) * t2
                + (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
        if len(points) < 4:
            return points
        result = []
        for i in range(1, len(points) - 2):
            p0, p1, p2, p3 = [np.array(p, dtype=np.float32) for p in points[i-1:i+3]]
            for t in np.linspace(0, 1, resolution):
                result.append(tuple(catmull_rom_spline(p0, p1, p2, p3, t)))
        return result

    def smooth_centers(self, centers, alpha=0.2):
        if self.prev_centers is None:
            self.prev_centers = centers
            return centers
        smoothed = []
        for i, curr in enumerate(centers):
            if i < len(self.prev_centers):
                prev = self.prev_centers[i]
                smoothed.append((
                    int(prev[0] * (1 - alpha) + curr[0] * alpha),
                    int(prev[1] * (1 - alpha) + curr[1] * alpha)
                ))
            else:
                smoothed.append(curr)
        self.prev_centers = smoothed
        return smoothed

    def process_frame(self, frame, params):
        """Process a single frame with blob detection"""
        # Convert frame to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Create threshold image (you might want to add adaptive thresholding)
        _, thresh = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY)
        
        out_h, out_w = frame.shape[:2]
        
        # Parse color parameters
        outline_col = self.parse_color(params.get('outline_color', '255,255,255'))
        alpha = params.get('alpha', 1.0)
        min_area = params.get('min_area', 100)
        max_area = params.get('max_area', 10000)
        max_blobs = params.get('max_blobs', 100)
        show_ids = params.get('show_ids', False)
        show_xy = params.get('show_xy', False)
        draw_trails = params.get('draw_trails', True)
        smooth_trails = params.get('smooth_trails', False)
        max_line_length_norm = params.get('max_line_length', 1.0)
        blob_thickness = params.get('blob_thickness', 2)

        # Set up blob detector
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.minArea = min_area
        detector_params.maxArea = max_area
        detector_params.filterByCircularity = False
        detector_params.filterByConvexity = False
        detector_params.filterByInertia = False
        detector_params.minThreshold = 1
        detector_params.maxThreshold = 255

        detector = cv2.SimpleBlobDetector_create(detector_params)
        keypoints = detector.detect(thresh)
        keypoints = sorted(keypoints, key=lambda kp: kp.size, reverse=True)[:max_blobs]

        # Create output image
        out_img = frame.copy().astype(np.float32) / 255.0
        centers = []

        # Process each blob
        for idx, kp in enumerate(keypoints):
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)

            w_box = h_box = size
            x = int(cx - size / 2)
            y = int(cy - size / 2)

            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w_box, out_w), min(y + h_box, out_h)

            centers.append((cx, cy))

            # Draw blob outline
            color_normalized = tuple(c/255.0 for c in outline_col)
            if blob_thickness == -1:
                cv2.rectangle(out_img, (x0, y0), (x1, y1), color_normalized, -1)
            else:
                cv2.rectangle(out_img, (x0, y0), (x1, y1), color_normalized, blob_thickness)

            # Draw text if requested
            if show_ids:
                if show_xy:
                    x_txt = cx / out_w
                    y_txt = cy / out_h
                    text = f"x:{x_txt:.3f} y:{y_txt:.3f}"
                else:
                    text = f"ID {idx}"

                font_scale = max(0.25, min(0.4, h_box / 100))
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                tx = x0
                ty = y0 - 2
                ty = max(ty, th)
                cv2.putText(out_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (1, 1, 1), 1)

        # Draw trails
        if draw_trails and len(centers) >= 2:
            trail_pts = centers
            if smooth_trails and len(trail_pts) >= 4:
                trail_pts = self.interpolate_catmull_rom(trail_pts, resolution=12)

            # Calculate trail length
            full_length = 0.0
            for i in range(len(trail_pts) - 1):
                p1 = np.array(trail_pts[i])
                p2 = np.array(trail_pts[i + 1])
                full_length += np.linalg.norm(p1 - p2)

            visible_length = full_length * np.clip(max_line_length_norm, 0.0, 1.0)

            # Trim trail to visible length
            trimmed = []
            total = 0.0
            for i in range(len(trail_pts) - 1, 0, -1):
                p1 = np.array(trail_pts[i])
                p2 = np.array(trail_pts[i - 1])
                segment = np.linalg.norm(p1 - p2)
                total += segment
                trimmed.insert(0, trail_pts[i])
                if total >= visible_length:
                    trimmed.insert(0, trail_pts[i - 1])
                    break

            if len(trimmed) >= 2:
                pts = np.array(trimmed, dtype=np.int32).reshape((-1, 1, 2))
                trail_color = self.parse_color(params.get('trail_color', '255,255,255'))
                trail_thickness = params.get('trail_thickness', 2)
                cv2.polylines(out_img, [pts], isClosed=False, 
                             color=tuple(c/255.0 for c in trail_color), thickness=trail_thickness)

        # Convert back to uint8
        out_img = (out_img * 255).astype(np.uint8)
        return out_img

    def parse_color(self, color_str):
        """Parse color string 'R,G,B' to tuple"""
        try:
            return tuple(map(int, color_str.split(',')))
        except:
            return (255, 255, 255)  # Default white

    def process_image(self, input_path, output_path, params):
        """Process a single image"""
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Could not load image")
        
        processed = self.process_frame(img, params)
        cv2.imwrite(output_path, processed)
        return output_path

    def process_video(self, input_path, output_path, params):
        """Process a video file"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame, params)
            out.write(processed_frame)
            frame_count += 1

        cap.release()
        out.release()
        return output_path, frame_count
