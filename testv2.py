import cv2
import numpy as np
import pickle
import threading
import time

    

def undistort_image(image, calibration_data):
    K = calibration_data['K']
    D = calibration_data['D']
    h, w = image.shape[:2]
 
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )

    undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_img


def generate_lut(world_x_min, world_x_max, world_x_interval,
                 world_y_min, world_y_max, world_y_interval,
                 K, extrinsic_matrix):
    world_x_coords = np.arange(world_x_min, world_x_max, world_x_interval)
    world_y_coords = np.arange(world_y_min, world_y_max, world_y_interval)

    output_height = len(world_y_coords)
    output_width = len(world_x_coords)

    xv, yv = np.meshgrid(world_x_coords, world_y_coords)
    ones = np.ones_like(xv)
    zeros = np.zeros_like(xv)

    world_points = np.stack((xv, yv, zeros, ones), axis=-1).reshape(-1, 4).T  # shape: (4, N)

    # ??? ???? ??
    camera_points = extrinsic_matrix @ world_points  # shape: (3, N)

    # ??? ???? ??
    image_points = K @ camera_points  # shape: (3, N)
    image_points /= image_points[2, :]  # ???

    u_coords = image_points[0, :].reshape((output_height, output_width))
    v_coords = image_points[1, :].reshape((output_height, output_width))

    map_x = u_coords.astype(np.float32)
    map_y = v_coords.astype(np.float32)

    return map_x, map_y

def generate_bev_image(image, map_x, map_y):
    bev_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return bev_image

def crop_polygon(image, points, angle = 0):

    hull = cv2.convexHull(np.array(points, dtype=np.int32))

    # Step 2: ??? ??
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Step 3: ??? ?? ????
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    # Step 4: Bounding Rect? ???? ??? ???
    x, y, w, h = cv2.boundingRect(hull)
    cropped_image = cropped_image[y:y+h, x:x+w]

    # Step 5: ??? ??
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return rotated_image

def calculate_pixel_distance(image, point, idx,low_threshold=50, high_threshold=100,
                             min_line_length=10, max_line_gap=10):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    cv2.imwrite(f'edge{idx}.jpg', edges)

    # Step 2: Hough Line Transform?? ?? ??
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        # ?? ? ?? ??
        max_length = 0
        thickest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                thickest_line = (x1, y1, x2, y2)

        # Step 3: ??? ????? ?? ??
        x0, y0 = point
        x1, y1, x2, y2 = thickest_line

        # ?? ??? ?? ?? (Ax + By + C = 0)
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # ?? ?? ??? ?? ?? (?? ??)
        distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)

        return distance
    else:
        print("No lines detected.")
        return None, None    

def process_camera(idx, device, calibration_data, bev_results, lock, config, roi, point):
    # 1. ??? ??
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"Cannot open camera {device}")
        return
    ret, img = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to grab frame from camera {device}")
        return

    # 2. ??? ?? ??
    undistorted_img = undistort_image(img, calibration_data)

    # 3. LUT ?? (?? ??? ??? ? ???? ?? ??)
    world_x_min = config['world_x_min']
    world_x_max = config['world_x_max']
    world_y_min = config['world_y_min']
    world_y_max = config['world_y_max']
    world_x_interval = config['world_x_interval']
    world_y_interval = config['world_y_interval']

    # LUT ??
    map_x, map_y = generate_lut(world_x_min, world_x_max, world_x_interval,
                                world_y_min, world_y_max, world_y_interval,
                                calibration_data['K'], calibration_data['extrinsic_matrix'])

    # 4. BEV ??? ??
    bev_image = generate_bev_image(undistorted_img, map_x, map_y)
    cv2.imwrite(f'bev_image{idx}.jpg', bev_image)
    roi_image = crop_polygon(bev_image, roi)
    cv2.imwrite(f'roi_image{idx}.jpg',roi_image)
    distance = calculate_pixel_distance(roi_image, point, idx=idx)

    # ?? ??
    with lock:
        bev_results[idx] = roi_image
        distances[idx] = distance


if __name__ == '__main__':
    # ??? ?? (? ???? ?? ??? ?? ??)
    camera_configs = [
        {'index': 0, 'device': '/dev/video0',
         'world_x_min': -0.0, 'world_x_max': 0.3,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005},
    
        {'index': 1, 'device': '/dev/video4',
         'world_x_min': -0.10, 'world_x_max': 0.2,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005},
        
        {'index': 2, 'device': '/dev/video8',
         'world_x_min': -0.1, 'world_x_max': 0.3,
         'world_y_min': -0.35, 'world_y_max': 0.35,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005},
        
        {'index': 3, 'device': '/dev/video25',
         'world_x_min': -0.15, 'world_x_max': 0.25,
         'world_y_min':-0.15, 'world_y_max': 0.3,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005}
    ]
    
    roi_config = [
        [(135, 43), (370, 194),  (324, 486), (48, 598),],
        [(97, 125), (47, 505), (450, 86), (344, 592)],
        [(107, 681), (107, 784), (795, 695), (795, 800)],
        [(203, 258), (203, 506), (491, 116), (463, 685)]
        ]
    

    distance_point = [(296,306), (25,247), (6,40), (15,15)]

    # ?????? ??? ??
    calibration_data = {}
    for config in camera_configs:
        idx = config['index']
        with open(f'calibration_data_camera{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            calibration_data[idx] = data  # 'K', 'D', 'extrinsic_matrix' ??

    # ???? ???? ? ??? ??
    threads = []
    bev_results = {}
    distances = {} 
    lock = threading.Lock()

    for config in camera_configs:
        idx = config['index']
        device = config['device']
        calib_data = calibration_data[idx]
        roi = roi_config[idx]
        point = distance_point[idx]
        thread = threading.Thread(target=process_camera, args=(idx, device, calib_data, bev_results, lock, config, roi, point))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for idx in distances:
        print(f'camera {idx} :', distances[idx])

