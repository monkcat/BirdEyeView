import cv2
import numpy as np
import pickle
import threading
import time


def capture_frame(camera_index, results, lock):

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return

    time.sleep(0.1) 

    ret, frame = cap.read()
    if ret:
        with lock:
            results[camera_index] = frame
    else:
        print(f"Failed to grab frame from camera {camera_index}")
    cap.release()

def capture_images():

    camera_indices = ['/dev/video0', '/dev/video4', '/dev/video8', '/dev/video25']
    threads = []
    results = {}
    lock = threading.Lock()

    # ? ???? ?? ????? ??
    for idx in camera_indices:
        thread = threading.Thread(target=capture_frame, args=(idx, results, lock))
    
        threads.append(thread)
        thread.start()

    # ?? ???? ??? ??? ??? ??
    for thread in threads:
        thread.join()

    # ??? ??? ??? ???? ???? ??
    return [results[idx] for idx in camera_indices if idx in results]

def load_calibration_data(camera_indices):
    calibration_data = {}
    for idx in camera_indices:
        with open(f'calibration_data_camera{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            calibration_data[idx] = {
                'K': data['K'],
                'D': data['D'],
            }
    return calibration_data

def undistort_images(images, calibration_data):
    undistorted_images = []
    for idx, img in enumerate(images):
        data = calibration_data[idx]
        K = data['K']
        D = data['D']
        h, w = img.shape[:2]

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
        )

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )

        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        undistorted_images.append(undistorted_img)
    return undistorted_images

    
def create_bev(calibration_data):
    camera_indices = [0, 1, 2, 3]

    # 1. ??? ??
    images = capture_images()
    if images is None:
        print("???? ??? ? ????.")
        return None
    
    undistorted_images = undistort_images(images, calibration_data)

    camera_indices = [0, 1, 2, 3]
    
    # ? ???? ?? ?? ?? (???? ? ??? ??)
    
    source_points_list = [
        np.float32([[6, 213], [632, 196], [26, 476], [629, 441]]),  # ??? 0
        np.float32([[53, 210], [594, 197], [78, 472], [629, 432]]),  # ??? 1
        np.float32([[112, 244], [565, 237], [34, 474], [581, 466]]),  # ??? 2
        np.float32([[154, 214], [624, 228], [108, 474], [619, 470]])   # ??? 3
    ]

    # ??? ??: ???? ??? ??
    destination_points_list = [
        np.float32([[0, 700], [0, 400], [300, 700], [300, 400]]),   # ??? 0
        np.float32([[1000, 400], [1000, 700], [700, 400], [700, 700]]),  # ??? 1
        np.float32([[465, 0], [765, 0], [465, 300], [765, 300]]),   # ??? 2
        np.float32([[235, 0], [535, 0], [235, 300], [535, 300]])    # ??? 3
    ]
    '''
    source_points_list = [
        np.float32([[0, 0], [640, 0], [0, 480], [640, 480]]),  # ??? 0
        np.float32([[0, 0], [640, 0], [0, 480], [640, 480]]),  # ??? 1
        np.float32([[0, 0], [640, 0], [0, 480], [640, 480]]),  # ??? 2
        np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])   # ??? 3
    ]

    # ??? ??: ???? ??? ??
    destination_points_list = [
        np.float32([[0, 700], [0, 400], [300, 700], [300, 400]]),   # ??? 0
        np.float32([[1000, 400], [1000, 700], [700, 400], [700, 700]]),  # ??? 1
        np.float32([[465, 0], [765, 0], [465, 300], [765, 300]]),   # ??? 2
        np.float32([[235, 0], [535, 0], [235, 300], [535, 300]])    # ??? 3
    ]
    '''

    # ??? ?? ??
    canvas_width, canvas_height = 1000, 700
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # ? ???? ???? ??
    for idx, image_original in enumerate(undistorted_images):
        # ?? ??? ??
        if image_original is None:
            print(f"???? ??? ? ????: {image_original}")
            continue

        # ?? ??? ??? ?? ??
        pts1 = source_points_list[idx]
        pts2 = destination_points_list[idx]

        # ?? ?? ?? ??
        H = cv2.getPerspectiveTransform(pts1, pts2)

        # ?? ?? ??
        img_warped = cv2.warpPerspective(image_original, H, (canvas_width, canvas_height))

        # ??? ???? ???? ??
        canvas = cv2.add(canvas, img_warped)
        
    return canvas
    


if __name__ == '__main__':
    start_time = time.time()
    camera_indices = [0, 1, 2, 3]

    # ?????? ??? ??
    calibration_data = {}
    for idx in camera_indices:
        with open(f'calibration_data_camera{idx}.pkl', 'rb') as f:
            data = pickle.load(f)
            calibration_data[idx] = data
    
    while True:
        bev_image = create_bev(calibration_data)
        cv2.imshow('Bird\'s Eye View', bev_image)
        cv2.waitKey(1000)  # 5000??? = 5?
        cv2.destroyAllWindows()