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
'''
def calculate_pixel_distance(image, point, idx,low_threshold=50, high_threshold=100,
                             min_line_length=10, max_line_gap=10):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    cv2.imwrite(f'./image/edge{idx}.jpg', edges)

    # Step 2: Hough Line Transform?? ?? ??
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=10,
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
'''



def calculate_pixel_distance(image,roi_points, base_point, idx,low_threshold=50, high_threshold=100,
                             min_line_length=10, max_line_gap=10):
    if idx == 0:
        input_line = (7, 305, 348, 376)
    elif idx == 1:
        input_line = (15, 280, 327, 312)
    elif idx == 2:
        input_line = (7, 178, 661, 167)
    else:
        input_line = (2, 140, 585, 146)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_polygon = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_polygon], 255)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edges_in_roi = cv2.bitwise_and(edges, edges, mask=mask)


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    cv2.imwrite(f'./image/edge{idx}.jpg', edges)

    # Step 2: Hough Line Transform?? ?? ??
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=10,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)     

    # Step 5: ??? ??? ROI ? ?? ?? ?? ??
    intersections = []
    x1, y1, x2, y2 = input_line
    for line in lines:
        x3, y3, x4, y4 = line[0]

        # ?? ? ?? ??
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:  # ??? ??
            continue

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        # ??? ROI ??? ??? ??
        if cv2.pointPolygonTest(roi_polygon, (px, py), False) >= 0:
            intersections.append((px, py))

    if not intersections:
        print("No intersections found within ROI.")
        return None

    # Step 6: ?? ?? ??? ??? ?? ??
    closest_point = None
    min_distance = float("inf")
    bx, by = base_point
    for px, py in intersections:
        distance = np.hypot(px - bx, py - by)
        if distance < min_distance:
            min_distance = distance
            closest_point = (px, py)

    return min_distance 


'''
def calculate_pixel_distance(image, roi_points, base_point, idx, imwrite=False):
    # 원본 이미지를 복사하여 그림을 그릴 수 있도록 합니다.
    output_image = image.copy()

    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 가우시안 블러 적용
    blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)

    # 3. Canny 엣지 검출
    edges = cv2.Canny(blur, 100, 200)

    # 디버깅을 위한 엣지 이미지 저장
    if imwrite:
        cv2.imwrite(f'./image/edge{idx}.jpg', edges)

    # 4. 입력 선 설정 (각 카메라 인덱스에 따라 다름)
    if idx == 0:
        input_line = (7, 305, 348, 376)
    elif idx == 1:
        input_line = (15, 280, 327, 312)
    elif idx == 2:
        input_line = (7, 178, 661, 167)
    else:
        input_line = (2, 140, 585, 146)

    # 입력 선을 이미지에 그림
    cv2.line(output_image, (input_line[0], input_line[1]), (input_line[2], input_line[3]), (0, 255, 0), 2)

    # 5. ROI 마스크 생성
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_polygon = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_polygon], 255)

    # 6. ROI 내의 엣지만 남기기
    edges_in_roi = cv2.bitwise_and(edges, edges, mask=mask)

    # 7. Hough 선 변환을 사용하여 선 검출 (cv2.HoughLines 사용)
    lines = cv2.HoughLines(edges_in_roi, rho=1, theta=np.pi / 180, threshold=15)

    if lines is None:
        # 선이 검출되지 않은 경우 큰 값 반환
        print("No lines detected.")
        return 10000

    # 검출된 선들을 이미지에 그림
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 이미지 크기에 따라 충분히 긴 선분으로 표시
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 8. 입력 선을 rho-theta 형태로 변환
    x1, y1, x2, y2 = input_line
    dx = x2 - x1
    dy = y2 - y1
    theta_input = np.arctan2(dy, dx)
    rho_input = x1 * np.cos(theta_input) + y1 * np.sin(theta_input)

    # 9. 검출된 선들과 입력 선의 교점 계산
    intersections = []
    for line in lines:
        rho, theta = line[0]

        # 두 직선의 교점 계산
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_theta_input = np.sin(theta_input)
        cos_theta_input = np.cos(theta_input)

        denom = cos_theta_input * sin_theta - sin_theta_input * cos_theta
        if denom == 0:
            # 평행한 선
            continue

        x = (sin_theta * rho_input - sin_theta_input * rho) / denom
        y = (cos_theta_input * rho - cos_theta * rho_input) / denom

        # 교점이 ROI 내부에 있는지 확인
        if cv2.pointPolygonTest(roi_polygon, (x, y), False) >= 0:
            intersections.append((x, y))
            # 교점을 이미지에 그림
            cv2.circle(output_image, (int(x), int(y)), 5, (255, 0, 0), -1)

    if not intersections:
        # 교점이 없는 경우 큰 값 반환
        # print("No intersections found within ROI.")
        return 10000

    # 10. 기준 점(base_point)과 교점들 간의 최소 거리 계산
    min_distance = float("inf")
    bx, by = base_point
    for px, py in intersections:
        distance = np.hypot(px - bx, py - by)
        if distance < min_distance:
            min_distance = distance

    # if imwrite:
    cv2.imwrite(f'./image/output_image_{idx}_test.jpg', output_image)

    return min_distance
'''
    

def process_camera(idx, device, calibration_data, bev_results, lock, config, roi, point):
    # 1. ??? ??
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 8)
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
    cv2.imwrite(f'./image/undistorted_image{idx}.jpg', undistorted_img)


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
    cv2.imwrite(f'./image/bev_image{idx}.jpg', bev_image)
    roi_image = crop_polygon(bev_image, roi)
    cv2.imwrite(f'./image/roi_image{idx}.jpg',roi_image)
    distance = calculate_pixel_distance(roi_image,roi, point, idx=idx)

    # ?? ??
    with lock:
        bev_results[idx] = roi_image
        distances[idx] = distance


if __name__ == '__main__':
    # ??? ?? (? ???? ?? ??? ?? ??)
    camera_configs = [
        {'index': 0, 'device': '/dev/video0',
         'world_x_min': -0.1, 'world_x_max': 0.3,
         'world_y_min': -0.3, 'world_y_max': 0.3,
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
        [(522, 774), (522, 469),  (172, 246), (181, 929)],
        [(67, 137), (44, 496), (426, 4), (322, 593)],
        [(109, 575), (119, 875), (774, 865), (771, 544)],
        [(201, 259), (793, 249), (203, 507), (793, 531)]
        ]
    

    distance_point = [(285,378), (30,204), (26,368), (1,249)]

    # ?????? ??? ??
    calibration_data = {}
    for config in camera_configs:
        idx = config['index']
        with open(f'./params/calibration_data_camera{idx}_2.pkl', 'rb') as f:
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

