import cv2
import numpy as np
import pickle
from multiprocessing import Process, Queue, Lock
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

def process_camera(idx, device, calibration_data, roi, point, lut, frame_queue, result_queue, lock, imwrite = False):
    cap = cv2.VideoCapture(device)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 8)

    if not cap.isOpened():
        print(f"Camera {device} failed to open.")
        return

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        # Undistort the image
        undistort_img = undistort_image(img, calibration_data)

        # Bird?s Eye View transformation
        map_x, map_y = lut
        bev_image = cv2.remap(undistort_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if (imwrite):
            cv2.imwrite(f'bev_image{idx}.jpg', bev_image)

        # Crop ROI and calculate distance
        roi_image = crop_polygon(bev_image, roi)
        if (imwrite):
            cv2.imwrite(f'roi_image{idx}.jpg',roi_image)
        distance = calculate_pixel_distance(roi_image, point, imwrite = imwrite)

        # Send the result back to the main process
        with lock:
            result_queue.put((idx, distance))

def crop_polygon(image, points):
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    cropped_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(hull)
    return cropped_image[y:y+h, x:x+w]

def calculate_pixel_distance(image, point, imwrite = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    if (imwrite):
        cv2.imwrite(f'edge{idx}.jpg', edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
    if lines is not None:
        max_length = 0
        thickest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > max_length:
                max_length = length
                thickest_line = (x1, y1, x2, y2)
        x0, y0 = point
        x1, y1, x2, y2 = thickest_line
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return abs(A * x0 + B * y0 + C) / np.hypot(A, B)
    return None

if __name__ == '__main__':
    # Camera configurations
    camera_configs = [
        {'index': 0, 'device': '/dev/video0',
         'world_x_min': -0.0, 'world_x_max': 0.3,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.001, 'world_y_interval': 0.001},

        {'index': 1, 'device': '/dev/video4',
         'world_x_min': -0.10, 'world_x_max': 0.2,
         'world_y_min': -0.15, 'world_y_max': 0.15,
         'world_x_interval': 0.001, 'world_y_interval': 0.001},

        {'index': 2, 'device': '/dev/video8',
         'world_x_min': -0.1, 'world_x_max': 0.3,
         'world_y_min': -0.35, 'world_y_max': 0.35,
         'world_x_interval': 0.001, 'world_y_interval': 0.001},

        {'index': 3, 'device': '/dev/video25',
         'world_x_min': -0.15, 'world_x_max': 0.25,
         'world_y_min': -0.15, 'world_y_max': 0.3,
         'world_x_interval': 0.001, 'world_y_interval': 0.001}
    ]

    roi_config = [
        [(135, 43), (370, 194), (324, 486), (48, 598)],
        [(97, 125), (47, 505), (450, 86), (344, 592)],
        [(111, 598), (111, 874), (536, 294), (554, 1122)],
        [(203, 258), (203, 506), (491, 116), (463, 685)]
    ]
    roi_config = list(np.array(roi_config) // 2)
    print(roi_config)

    distance_point = [(296, 306), (25, 247), (15, 15), (15, 15)]
    distance_point = list(np.array(distance_point) // 2)

    # Load calibration data
    calibration_data = {}
    lut = {}
    for config in camera_configs:
        idx = config['index']
        with open(f'calibration_data_camera{idx}_2.pkl', 'rb') as f:
            calibration_data[idx] = pickle.load(f)

        # Generate LUT
        calib_data = calibration_data[idx]
        lut[idx] = generate_lut(
            config['world_x_min'], config['world_x_max'], config['world_x_interval'],
            config['world_y_min'], config['world_y_max'], config['world_y_interval'],
            calib_data['K'], calib_data['extrinsic_matrix']
        )

    # Queues for communication
    frame_queue = Queue(maxsize=3)
    result_queue = Queue()
    lock = Lock()

    # Start camera processes
    processes = []
    imwrite = True
    for config in camera_configs:
        idx = config['index']
        device = config['device']
        roi = roi_config[idx]
        point = distance_point[idx]
        p = Process(target=process_camera, args=(
            idx, device, calibration_data[idx], roi, point, lut[idx],
            frame_queue, result_queue, lock, imwrite
        ))
        processes.append(p)
        p.start()

    try:
        while True:
            # Measure time for each iteration
            start_time = time.time()
            results = {}

            # Collect results from all cameras
            for _ in range(len(camera_configs)):
                try:
                    idx, distance = result_queue.get(timeout=2)  # Wait for each result
                    results[idx] = distance
                except Exception:
                    print("Timeout waiting for camera result.")
                    continue

            # Print results
            for idx in sorted(results.keys()):
                if len(results) == 4:
                    print(f"Camera {idx}: Distance = {results[idx]}")

            # Measure and print elapsed time
            elapsed_time = time.time() - start_time
            print(f"Iteration time: {elapsed_time:.4f} seconds")

    except KeyboardInterrupt:
        print("Stopping processes...")

    # Stop processes
    for p in processes:
        p.terminate()
