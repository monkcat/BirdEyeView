import cv2
import numpy as np
import pickle
from multiprocessing import Process, Manager, Lock

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

    world_points = np.stack((xv, yv, np.zeros_like(xv), ones), axis=-1).reshape(-1, 4).T  # shape: (4, N)

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

def crop_polygon(image, points, angle=0):
    hull = cv2.convexHull(np.array(points, dtype=np.int32))

    # ??? ??
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # ???? ???? ??? ??
    cropped_image = cv2.bitwise_and(image, image, mask=mask)

    # Bounding Rect? ???? ??? ??
    x, y, w, h = cv2.boundingRect(hull)
    cropped_image = cropped_image[y:y+h, x:x+w]

    # ??? ??
    if angle != 0:
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        return rotated_image
    else:
        return cropped_image

def calculate_pixel_distance(image, point, low_threshold=50, high_threshold=150,
                             min_line_length=100, max_line_gap=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Hough Line Transform? ?? ? ??
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        # ?? ? ? ??
        max_length = 0
        thickest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > max_length:
                max_length = length
                thickest_line = (x1, y1, x2, y2)

        # ?? ? ??? ?? ??
        x0, y0 = point
        x1, y1, x2, y2 = thickest_line

        # ?? ??? ?? (Ax + By + C = 0)
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # ?? ? ??? ?? ?? (??? ?)
        distance = abs(A * x0 + B * y0 + C) / np.hypot(A, B)

        return distance
    else:
        # ?? ???? ?? ?? None ??
        return None

def process_camera(idx, device, calibration_data, distances, lock, config, roi, point, lut):
    # 1. ??? ??
    cap = cv2.VideoCapture(device)

    # ??? ?? ??
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # ??? ?? ???? FPS ??
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        return
    ret, img = cap.read()
    cap.release()
    if not ret:
        return

    # 2. ??? ?? ??
    undistort_img = undistort_image(img, calibration_data)

    # 3. BEV ??? ?? (?? ??? LUT ??)
    map_x, map_y = lut[idx]
    bev_image = cv2.remap(undistort_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 4. ROI ?? ? ?? ??
    roi_image = crop_polygon(bev_image, roi)
    distance = calculate_pixel_distance(roi_image, point)

    # ?? ??
    with lock:
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
         'world_y_min': -0.15, 'world_y_max': 0.3,
         'world_x_interval': 0.0005, 'world_y_interval': 0.0005}
    ]

    roi_config = [
        [(135, 43), (370, 194), (324, 486), (48, 598)],
        [(97, 125), (47, 505), (450, 86), (344, 592)],
        [(111, 598), (111, 874), (536, 294), (554, 1122)],
        [(203, 258), (203, 506), (491, 116), (463, 685)]
    ]

    distance_point = [(296, 306), (25, 247), (15, 15), (15, 15)]

    # ?????? ??? ??
    calibration_data = {}
    for config in camera_configs:
        idx = config['index']
        with open(f'calibration_data_camera{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            calibration_data[idx] = data  # 'K', 'D', 'extrinsic_matrix' ??

    # LUT ?? ??
    lut = {}
    for config in camera_configs:
        idx = config['index']
        calib_data = calibration_data[idx]
        world_x_min = config['world_x_min']
        world_x_max = config['world_x_max']
        world_y_min = config['world_y_min']
        world_y_max = config['world_y_max']
        world_x_interval = config['world_x_interval']
        world_y_interval = config['world_y_interval']

        map_x, map_y = generate_lut(world_x_min, world_x_max, world_x_interval,
                                    world_y_min, world_y_max, world_y_interval,
                                    calib_data['K'], calib_data['extrinsic_matrix'])
        lut[idx] = (map_x, map_y)

    # multiprocessing? ???? ? ??? ??
    manager = Manager()
    distances = manager.dict()
    lock = Lock()

    processes = []
    for config in camera_configs:
        idx = config['index']
        device = config['device']
        calib_data = calibration_data[idx]
        roi = roi_config[idx]
        point = distance_point[idx]
        p = Process(target=process_camera, args=(idx, device, calib_data, distances, lock, config, roi, point, lut))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # ?? ?? ??? ??
    pixel_distances = []
    for idx in sorted(distances.keys()):
        pixel_distances.append(distances[idx])

    # ?? ??
    print(pixel_distances)
