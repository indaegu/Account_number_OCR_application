import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import socket
import struct
import io
import re
import pymysql
import threading
import time

buff = []
connection = pymysql.connect(
    host="localhost",
    user="root",
    password="21812200",
    database="mydb1",
    charset='utf8'
)


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def send_thread(client_socket, client_address):
    client = client_socket
    # Receive image size

    image_size_data = recvall(client, 4)  # 전달된 데이터의 사이즈를 4바이트로 계산한다.
    image_size = struct.unpack('!i', image_size_data)[0]  # 계산한 사이즈를 정수형으로 변환

    # Receive image data
    image_data = recvall(client, image_size)  # 앞서 계산한 사이즈로 데이터를 전송받는다.
    image = Image.open(io.BytesIO(image_data))  # 이미지 객체를 생성
    image.save('received_image.jpg', 'JPEG')  # 이미지를 저장한다.

    ################################# 위의 경우에는 클라이언트의 접속을 확인하는 과정 #################################
    imag = cv2.imread('received_image.jpg')  # opencv를 사용하여 이미지 저장
    img = cv2.resize(imag, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # 리사이징으로 픽셀 수 를 증가시켜 인식률을 증가시킴
    height, width, channel = img.shape  # 이미지 사이즈 및 커널 정보를 저장
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 사진을 흑백화하여 인식률을 높힘.

    # plt.figure(figsize=(14, 10))
    # plt.subplot(2, 4, 1)
    # plt.imshow(gray, cmap="gray")

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)  # opencv를 이용하여 이미지를 가우시안 필터를 적용, 이미지 분석을 쉽도록 변형

    img_thresh = cv2.adaptiveThreshold(  # 적응형 threshold로 조명의 영향력을 대폭 감소시킴
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    # plt.subplot(2, 4, 2)
    # plt.imshow(img_thresh, cmap='gray')
    # plt.waitforbuttonpress()
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 검출
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 이미지와 동일한 사이즈의 검은 공간을 생성
    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))  # 외곽선 그리기

    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 4, 3)
    # plt.imshow(temp_result, cmap='gray')
    # plt.waitforbuttonpress()
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # 공간 초기화
    contours_dict = []

    for contour in contours:  # 윤곽선
        x, y, w, h = cv2.boundingRect(contour)  # 해당 윤곽선을 포함하는 최소한의 사각형을 생성
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
        # 사각형을 생성한다

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
        # 딕셔너리를 생성하여 기록한다. cx,cy는 중심좌표에 해당함

    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 4, 4)
    # plt.imshow(temp_result, cmap='gray')
    MIN_AREA = 100  # 계좌번호 윤곽선 최소 넓이 지정
    MAX_AREA = 2000  # 계좌번호 윤곽선 최소 넓이 지정
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 최소 너비 높이 범위 지정
    MIN_RATIO, MAX_RATIO = 0.1, 6.0  # 최소 비율 범위 지정

    possible_contours = []  # possible_contours에 저장

    cnt = 0
    for d in contours_dict:
        area = d['h'] * d['w']
        ratio = d['h'] / d['w']

        # 위에 설정한 범위의 조건을 비교, 맞추면서 다시한번 possible_contours에 저장해준다.
        # 각 윤곽선의 idx값을 매겨놓고, 나중에 조건에 맞는 윤곽선들의 idx만 따로 빼낼 것이다.  d['idx'] = cnt
        if MAX_AREA > area > MIN_AREA and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT and MIN_RATIO < ratio < MAX_RATIO:
            # 최소, 최대 범위를 넘기 않고, 최소한의 높이와 너비를 넘기며, 적절한 비율을 가진 경우
            d['idx'] = cnt  # count
            cnt += 1
            possible_contours.append(d)  # 계좌번호 가능성 있는 윤곽선을 기록

    # visualize possible contours
    # possible contours의 정렬방식을 보고 계좌번호 후보들을 추려낸다. 계좌번호은 어느정도 규칙적으로 일렬로 나타난다. 순차적,각도,배열모양..
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        # 문자 후보군들의 위치를 시각적으로 표현하기 위해 출력
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 4, 5)
    # plt.imshow(temp_result, cmap='gray')
    # plt.waitforbuttonpress()
    MAX_DIAG_MULTIPLYER = 14  # 대각선길이
    MAX_ANGLE_DIFF = 12.0  # i번째 contour와 i+1번째 contour 의 각도
    MAX_AREA_DIFF = 2  # 면적의 차이
    MAX_WIDTH_DIFF = 1.6  # 너비 차이
    MAX_HEIGHT_DIFF = 1  # 높이 차이
    MIN_N_MATCHED = 12

    # find_chars 함수로 지정한다. 나중에 재귀함수로 반복해서 찾기 위함이다. idx값 저장

    def find_chars(contour_list):
        matched_result_idx = []

        # 이중for문으로 첫번째 contour와 두번째 contour를 비교

        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:  # index가 동일한 경우 넘어간다.
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])
                # 두 윤곽선의 중앙좌표의 거리를 계산

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                # np.linalg.norm(a - b) 벡터 a와 벡터 b 사이의 거리를 구한다.
                # 삼각함수 사용

                distance = np.linalg.norm(
                    np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))  # i번째 윤곽선과 i+1번째 윤곽선의 거리를 계산

                if dx == 0:  # 계산 오류를 방지하기 위해서 각도를 지정
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))  # 각도 계산
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])  # 면적의 비율
                width_diff = abs(d1['w'] - d2['w']) / d1['w']  # 너비의 비율
                height_diff = abs(d1['h'] - d2['h']) / d1['h']  # 높이의 비율

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])
            # 각도와 면적, 너비의 비율이 적저란 경우, 해당 윤곽선을 i번째 윤곽선을 시작지점으로 하는 계좌번호 후보군으로 지정

            matched_contours_idx.append(d1['idx'])
            # 육관선 개수가 3보다 작을때는 continue를 통해 반복문 반복
            if len(matched_contours_idx) < MIN_N_MATCHED:  # 최소 계좌번호 길이를 충족하는지 확인한다
                continue
            # 최종후보군에 넣어주기
            matched_result_idx.append(matched_contours_idx)  #
            # 아직 확인하지 않은 것들을 다시 한번 비교하고 넣어준다.
            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])
            # 위에서 하나의 계좌번호 후보가 등록되었으나, 가능성이 남아있으면 나머지도 확인할 필요성이 있으므로 index를 정리, 재귀로 다시 실행함

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)
            # 재귀가 완료되면, 해당 재귀 결과를 입력함.

            break

        return matched_result_idx

    # 위의 과정은 면적, 너비, 높이의 비율을 적절히 조정하면 하나의 문자열만 등록됨. 이를 확인하고자, 위처럼 복잡한 방식으로 디자인 함

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255),
                          thickness=2)

    # plt.figure(figsize=(12, 10))
    # plt.subplot(2, 4, 6)
    # plt.imshow(temp_result, cmap='gray')

    ################################  계좌번호 각도를 조절하는 중  ###############################

    PLATE_WIDTH_PADDING = 1.5  # 1.3
    PLATE_HEIGHT_PADDING = 3  # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []
    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        # 센터 좌표 구하기
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2  # 계좌번호의 시작과 끝부분의 x좌표의 중앙값
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2  # 계좌번호의 시작과 끝부분의 y좌표의 중앙값

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        # 계좌번호의 너비를 계산, 약간의 너비 오차를 방지하기 위해서 패딩함

        sum_height = 0
        for d in sorted_chars:  # 계좌번호의 문자 평균 높이를 계산함
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        # 계좌번호의 기울어진 각도를 구하기 (삼각함수 이용)
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  # 계좌번호 높이
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        # 계좌의 시작과 끝부분의 벡터의 크기를 계산, 이를 빗변으로 활용함

        angle = np.degrees(np.arctan(triangle_height / triangle_hypotenus)) * 0.5
        # 삼각함수(탄젠트)를 활용하여 기울기를 계산

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
        # 계산한 각도와 중앙값을 사용하여 사진의 기울기를 조절
        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
        # plt.subplot(len(matched_result), 1, i + 1)
        # plt.subplot(2, 4, 7)
        # plt.imshow(img_cropped, cmap='gray')
        # plt.waitforbuttonpress()

    try:
        extracted_text = pytesseract.image_to_string(img_cropped, lang='kor', )
    except Exception as e:
        data_to_send = "error"
        print("사진이 이상합니다.")
        client.sendall(data_to_send.encode())
        client.close()
        del client

    filtered_text = re.sub(r'[^가-힣0-9\s]', '', extracted_text)
    letters = ''.join(re.findall('[가-힣]+', filtered_text))
    numbers = ''.join(re.findall('\d+', filtered_text))
    num_leg = len(numbers)
    print(f"은행명 : '{letters}', 계좌번호 : '{numbers}', 계좌길이 : '{num_leg}'")

    cursor = connection.cursor()

    sql_query = f"select bank_info.계좌길이 from bank_info where bank_info.은행명='{letters}'"
    cursor.execute(sql_query)
    results = cursor.fetchall()

    if (num_leg != results[0][0] or results == ()):  # 길이가 맞지 않는 경우
        print(results[0][0])
        data_to_send = "error,leng"
        print(data_to_send, " 길이가 부적절합니다.")
        client.sendall(data_to_send.encode())
        client.close()
        del client
    else:
        sql_query = f"""SELECT bank_account.계좌번호, bank_info.은행명, customer_info.고객명
                                           FROM bank_account
                                           INNER JOIN bank_info ON bank_account.은행ID = bank_info.은행ID
                                           INNER JOIN customer_info ON bank_account.고객ID = customer_info.고객ID
                                           WHERE bank_account.계좌번호 = '{numbers}' AND bank_info.은행명 = '{letters}'"""
        cursor.execute(sql_query)
        results = cursor.fetchall()
        for result in results:
            print(f"계좌번호: {result[0]}, 은행명: {result[1]}, 고객명: {result[2]}")

        if (results == ()):  # 길이가 확인된 상태
            print(results)
            data_to_send = "error"
            print(data_to_send)
            client.sendall(data_to_send.encode())
            client.close()
            del client
        else:
            data_to_send = ','.join(results[0])
            # print(data_to_send)
            client.sendall(data_to_send.encode())
            print("send.")
            client.close()
            del client


def recv_thread():
    while True:
        client_socket, client_address = server_socket.accept()  # 클라이언트 정보를 저장
        print(f"Connection from {client_address}")  # 클라이언트의 IP를 확인
        client_thread = threading.Thread(target=send_thread, args=(client_socket, client_address))
        client_thread.start()


server_ip = '0.0.0.0'
server_port = 46460  # 서버의 포트번호

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 서버를 TCP로 생성
server_socket.bind((server_ip, server_port))  # 소켓 설정 ip=192.168.72.169
server_socket.listen()  # 접촉을 허용함

thread1 = threading.Thread(target=recv_thread)
thread1.start()
print("Server is ready to receive images...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down the server...")
    connection.close()
    server_socket.close()
