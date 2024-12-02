import cv2
import numpy as np

# ?? ??
points = []  # ???? ??? ? ??

def mouse_callback(event, x, y, flags, param):
    """
    ??? ?? ??? ??? ??? ??
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # ?? ?? ??
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

def crop_and_rotate(image, points, angle):
    """
    ??? ??? ????, ??? ??? ??

    Parameters:
    image (numpy.ndarray): ?? ???
    points (list): ??? ??? ???
    angle (float): ?? ?? (??? ??)

    Returns:
    numpy.ndarray: ???? ??? ???
    """
    # Step 1: Convex Hull ??
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

if __name__ == "__main__":
    # ?? ??? ????
    image = cv2.imread("./image/roi_image3.jpg")  # BEV ???? ??
    clone = image.copy()  # ?? ???? ???? ?? ??

    # ??? ?? ??
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", mouse_callback)

    print("????? ??? ?? ?????. ?? ? 's'? ?????.")

    while True:
        # ??? ? ??
        for point in points:
            cv2.circle(clone, point, 5, (0, 0, 255), -1)

        cv2.imshow("Select Points", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):  # 's' ?? ??? ?? ??
            break
        elif key == ord("r"):  # 'r' ?? ??? ???
            points = []
            clone = image.copy()
            print("Points reset.")

    # ??? ?? ???? ?? ???? ? ??
    if len(points) > 2:  # ?? 3?? ?? ??
        rotated_image = crop_and_rotate(image, points, angle=11)  # 20? ??? ??

        # ?? ??? ?? ? ??
        cv2.imshow("Cropped and Rotated Image", rotated_image)
        cv2.imwrite("cropped_and_rotated.jpg", rotated_image)
        cv2.waitKey(0)

    else:
        print("?? 3?? ?? ???? ???.")

    cv2.destroyAllWindows()
