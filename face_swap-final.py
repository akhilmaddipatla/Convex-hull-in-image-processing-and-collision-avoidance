import cv2
import numpy as np
import dlib
import time


def index_extract(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


image = cv2.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/Akhil 4.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(image_gray)
image2 = cv2.imread("C:/Users/makhi/OneDrive/Documents/Project - 1/images/Robert_Pattinson.jpg")
image2_S = cv2.resize(image2, (960, 540))
image2_gray = cv2.cvtColor(image2_S, cv2.COLOR_BGR2GRAY)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
height, width, channels = image2_S.shape
image2_new_face = np.zeros((height, width, channels), np.uint8)




# Face 1
faces = detector(image_gray)
for face in faces:
    landmarks = predictor(image_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))



    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(image, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)

    face_image_1 = cv2.bitwise_and(image, image, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    sub_div = cv2.Subdiv2D(rect)
    sub_div.insert(landmarks_points)
    triangles = sub_div.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])


        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = index_extract(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = index_extract(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = index_extract(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)



# Face 2
faces2 = detector(image2_gray)
for face in faces2:
    landmarks = predictor(image2_gray, face)
    landmarks_points2 = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points2.append((x, y))


    points2 = np.array(landmarks_points2, np.int32)
    convexhull2 = cv2.convexHull(points2)

lines_space_mask = np.zeros_like(image_gray)
lines_space_new_face = np.zeros_like(image2_S)
# Triangulation of both faces
for triangle_index in indexes_triangles:
    # Triangulation of the first face
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = image[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)


    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

    # Lines space
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(image, image, mask=lines_space_mask)

    # Triangulation of second face
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2

    cropped_tr2_mask = np.zeros((h, w), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    # Reconstructing destination face
    image2_new_face_rect_area = image2_new_face[y: y + h, x: x + w]
    image2_new_face_rect_area_gray = cv2.cvtColor(image2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(image2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    image2_new_face_rect_area = cv2.add(image2_new_face_rect_area, warped_triangle)
    image2_new_face[y: y + h, x: x + w] = image2_new_face_rect_area



# Face swapped (putting 1st face into 2nd face)
image2_face_mask = np.zeros_like(image2_gray)
image2_head_mask = cv2.fillConvexPoly(image2_face_mask, convexhull2, 255)
image2_face_mask = cv2.bitwise_not(image2_head_mask)


image2_head_noface = cv2.bitwise_and(image2_S, image2_S, mask=image2_face_mask)
result = cv2.add(image2_head_noface, image2_new_face)

(x, y, w, h) = cv2.boundingRect(convexhull2)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

seamlessclone = cv2.seamlessClone(result, image2_S, image2_head_mask, center_face2, cv2.NORMAL_CLONE)

cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)



cv2.destroyAllWindows()