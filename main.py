# import cv2
# import torch
# import numpy as np
# import pyttsx3
# import threading
# import time
# from ultralytics import YOLO
#
# # -------------------------
# # TTS (thread bilan - parallel)
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
#
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run).start()  # 🔥 PARALLEL: ovoz alohida thread
#
# # -------------------------
# # MODEL LOAD
# model = YOLO("yolov8n.pt")
#
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas.eval()
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# # -------------------------
# cap = cv2.VideoCapture(0)
#
# # -------- SEQUENTIAL FUNCTION --------
# def process_sequential(frame):
#     start = time.time()
#
#     # 1. YOLO (ketma-ket)
#     results = model(frame, verbose=False)[0]
#
#     # 2. MiDaS (YOLOdan keyin ishlaydi)
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img)
#
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#
#     depth_map = prediction.cpu().numpy()
#
#     elapsed = time.time() - start
#     return results, depth_map, elapsed
#
#
# # -------- PARALLEL FUNCTION --------
# def process_parallel(frame):
#     start = time.time()
#
#     results = [None]
#     depth_map = [None]
#
#     # 🔥 THREAD 1: YOLO
#     def run_yolo():
#         results[0] = model(frame, verbose=False)[0]
#
#     # 🔥 THREAD 2: MiDaS
#     def run_midas():
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = transform(img)
#
#         with torch.no_grad():
#             prediction = midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
#
#         depth_map[0] = prediction.cpu().numpy()
#
#     # 🔥 PARALLEL START
#     t1 = threading.Thread(target=run_yolo)
#     t2 = threading.Thread(target=run_midas)
#
#     t1.start()
#     t2.start()
#
#     # 🔥 WAIT (join)
#     t1.join()
#     t2.join()
#
#     elapsed = time.time() - start
#     return results[0], depth_map[0], elapsed
#
#
# # -------------------------
# mode = "parallel"  # "sequential" yoki "parallel"
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # MODE TANLASH
#     if mode == "sequential":
#         results, depth_map, elapsed = process_sequential(frame)
#     else:
#         results, depth_map, elapsed = process_parallel(frame)
#
#     # NATIJA CHIQARISH
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         depth_value = depth_map[cy, cx]
#
#         if depth_value < 80:
#             speak(f"Diqqat! {label} yaqin!")
#
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#         cv2.putText(frame, label, (x1,y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
#
#     # ⏱ VAQT KO‘RSATISH
#     cv2.putText(frame, f"Mode: {mode}", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)
#
#     cv2.putText(frame, f"Time: {round(elapsed,3)} s", (20,70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)
#
#     cv2.imshow("Compare System", frame)
#
#     key = cv2.waitKey(1)
#
#     if key == ord('q'):
#         break
#     elif key == ord('s'):
#         mode = "sequential"
#     elif key == ord('p'):
#         mode = "parallel"
#
# cap.release()
# cv2.destroyAllWindows()


# 2

# import cv2
# import torch
# import numpy as np
# import pyttsx3
# import threading
# import time
# from ultralytics import YOLO
#
# # -------------------------
# # TTS (parallel)
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
#
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run).start()  # 🔥 parallel ovoz
#
# # -------------------------
# # MODEL LOAD
# model = YOLO("yolov8n.pt")
#
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas.eval()
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# # -------------------------
# cap = cv2.VideoCapture(0)
#
# # -------- SEQUENTIAL --------
# def process_sequential(frame):
#     start = time.time()
#
#     # YOLO
#     results = model(frame, verbose=False)[0]
#
#     # MiDaS
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img)
#
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#
#     depth_map = prediction.cpu().numpy()
#
#     elapsed = time.time() - start
#     return results, depth_map, elapsed
#
#
# # -------- PARALLEL --------
# def process_parallel(frame):
#     start = time.time()
#
#     results = [None]
#     depth_map = [None]
#
#     # 🔥 YOLO thread
#     def run_yolo():
#         results[0] = model(frame, verbose=False)[0]
#
#     # 🔥 MiDaS thread
#     def run_midas():
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = transform(img)
#
#         with torch.no_grad():
#             prediction = midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
#
#         depth_map[0] = prediction.cpu().numpy()
#
#     # 🔥 parallel start
#     t1 = threading.Thread(target=run_yolo)
#     t2 = threading.Thread(target=run_midas)
#
#     t1.start()
#     t2.start()
#
#     t1.join()
#     t2.join()
#
#     elapsed = time.time() - start
#     return results[0], depth_map[0], elapsed
#
#
# # -------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 🔥 IKKALA USULNI HAM HISOBLAYMIZ
#     results_seq, depth_seq, time_seq = process_sequential(frame)
#     results_par, depth_par, time_par = process_parallel(frame)
#
#     # 🔥 parallel natijani ko‘rsatamiz
#     results = results_par
#     depth_map = depth_par
#
#     # NATIJA CHIZISH
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         depth_value = depth_map[cy, cx]
#
#         if depth_value < 80:
#             speak(f"Diqqat! {label} yaqin!")
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, label, (x1, y1-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
#
#     # ⏱ VAQT HISOBLASH
#     speedup = time_seq / time_par if time_par > 0 else 0
#
#     cv2.putText(frame, f"Sequential: {round(time_seq,3)} s", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
#
#     cv2.putText(frame, f"Parallel: {round(time_par,3)} s", (20,70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#
#     cv2.putText(frame, f"Speedup: {round(speedup,2)}x", (20,100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
#
#     cv2.imshow("Sequential vs Parallel", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()




# 3




# import cv2
# import torch
# import numpy as np
# import pyttsx3
# import threading
# import time
# from ultralytics import YOLO
#
# # -------------------------
# # TTS (parallel)
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
#
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run).start()
#
# # -------------------------
# # MODEL LOAD
# model = YOLO("yolov8n.pt")
#
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas.eval()
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# # -------------------------
# cap = cv2.VideoCapture(0)
#
# # -------- SEQUENTIAL --------
# def process_sequential(frame):
#     start = time.time()
#
#     results = model(frame, verbose=False)[0]
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img)
#
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#
#     depth_map = prediction.cpu().numpy()
#
#     elapsed = time.time() - start
#     return results, depth_map, elapsed
#
#
# # -------- PARALLEL --------
# def process_parallel(frame):
#     start = time.time()
#
#     results = [None]
#     depth_map = [None]
#
#     def run_yolo():
#         results[0] = model(frame, verbose=False)[0]
#
#     def run_midas():
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = transform(img)
#
#         with torch.no_grad():
#             prediction = midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
#
#         depth_map[0] = prediction.cpu().numpy()
#
#     t1 = threading.Thread(target=run_yolo)
#     t2 = threading.Thread(target=run_midas)
#
#     t1.start()
#     t2.start()
#
#     t1.join()
#     t2.join()
#
#     elapsed = time.time() - start
#     return results[0], depth_map[0], elapsed
#
#
# # -------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 🔥 IKKALA USUL
#     results_seq, depth_seq, time_seq = process_sequential(frame)
#     results_par, depth_par, time_par = process_parallel(frame)
#
#     # parallel natijani ishlatamiz
#     results = results_par
#     depth_map = depth_par
#
#     # NATIJA
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         depth_value = depth_map[cy, cx]
#
#         # 🔥 MASOFA (approx)
#         distance = 1000 / (depth_value + 1e-6)
#
#         # 🔥 XAVF DARAJASI + RANG
#         if distance < 2:
#             color = (0, 0, 255)   # qizil
#             status = "Juda yaqin"
#             speak(f"Diqqat! {label} juda yaqin!")
#         elif distance < 5:
#             color = (0, 255, 255) # sariq
#             status = "O‘rtacha"
#         else:
#             color = (0, 255, 0)   # yashil
#             status = "Uzoq"
#
#         # 🔥 CHIZISH
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, f"{label} {round(distance,1)}m",
#                     (x1, y1-25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#         cv2.putText(frame, status,
#                     (x1, y1-5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     # ⏱ VAQT
#     speedup = time_seq / time_par if time_par > 0 else 0
#
#     cv2.putText(frame, f"Sequential: {round(time_seq,3)} s", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
#
#     cv2.putText(frame, f"Parallel: {round(time_par,3)} s", (20,70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#
#     cv2.putText(frame, f"Speedup: {round(speedup,2)}x", (20,100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
#
#     cv2.imshow("AI System (Distance + Parallel)", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()




# 4



# import cv2
# import torch
# import numpy as np
# import pyttsx3
# import threading
# import time
# from ultralytics import YOLO
#
# # -------------------------
# # TTS (parallel ovoz)
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
#
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run).start()
#
# # -------------------------
# # MODEL LOAD
# model = YOLO("yolov8n.pt")
#
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas.eval()
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# # -------------------------
# cap = cv2.VideoCapture(0)
#
# # -------- SEQUENTIAL --------
# def process_sequential(frame):
#     start = time.time()
#
#     results = model(frame, verbose=False)[0]
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img)
#
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#
#     depth_map = prediction.cpu().numpy()
#
#     elapsed = time.time() - start
#     return results, depth_map, elapsed
#
#
# # -------- PARALLEL --------
# def process_parallel(frame):
#     start = time.time()
#
#     results = [None]
#     depth_map = [None]
#
#     def run_yolo():
#         results[0] = model(frame, verbose=False)[0]
#
#     def run_midas():
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = transform(img)
#
#         with torch.no_grad():
#             prediction = midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze()
#
#         depth_map[0] = prediction.cpu().numpy()
#
#     t1 = threading.Thread(target=run_yolo)
#     t2 = threading.Thread(target=run_midas)
#
#     t1.start()
#     t2.start()
#
#     t1.join()
#     t2.join()
#
#     elapsed = time.time() - start
#     return results[0], depth_map[0], elapsed
#
#
# # -------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 🔥 IKKALA HOLAT
#     results_seq, depth_seq, time_seq = process_sequential(frame)
#     results_par, depth_par, time_par = process_parallel(frame)
#
#     results = results_par
#     depth_map = depth_par
#
#     # -------------------------
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         depth_value = depth_map[cy, cx]
#
#         # 🔥 MASOFA HISOBLASH
#         distance = 1000 / (depth_value + 1e-6)
#
#         # 🔥 CHIROYLI FORMAT (yaxlitlash)
#         if distance < 10:
#             dist_text = f"{distance:.1f}m"
#         else:
#             dist_text = f"{int(distance)}m"
#
#         # -------------------------
#         # XAVF DARAJASI
#         if distance < 2:
#             color = (0, 0, 255)
#             status = "Juda yaqin"
#             speak(f"Diqqat! {label} juda yaqin!")
#         elif distance < 5:
#             color = (0, 255, 255)
#             status = "O‘rtacha"
#         else:
#             color = (0, 255, 0)
#             status = "Uzoq"
#
#         # -------------------------
#         # CHIZISH
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#
#         cv2.putText(frame, f"{label} {dist_text}",
#                     (x1, y1 - 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#         cv2.putText(frame, status,
#                     (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     # -------------------------
#     # VAQT + SPEEDUP
#     speedup = time_seq / time_par if time_par > 0 else 0
#
#     cv2.putText(frame, f"Sequential: {round(time_seq,3)} s", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
#
#     cv2.putText(frame, f"Parallel: {round(time_par,3)} s", (20,70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#
#     cv2.putText(frame, f"Speedup: {round(speedup,2)}x", (20,100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
#
#     cv2.imshow("AI Vision System (Improved)", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# 5


import cv2
import torch
import numpy as np
import pyttsx3
import threading
import time
from ultralytics import YOLO

# -------------------------
# 🔊 TEXT TO SPEECH (parallel)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# -------------------------
# 🔥 SPEECH CONTROL (spam oldini olish)
last_spoken = {}
COOLDOWN = 3  # sekund

# -------------------------
# MODEL LOAD
model = YOLO("yolov8n.pt")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# -------------------------
cap = cv2.VideoCapture(0)

# =========================
# SEQUENTIAL
# =========================
def process_sequential(frame):
    start = time.time()

    results = model(frame, verbose=False)[0]

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    elapsed = time.time() - start
    return results, depth_map, elapsed


# =========================
# PARALLEL
# =========================
def process_parallel(frame):
    start = time.time()

    results = [None]
    depth_map = [None]

    def run_yolo():
        results[0] = model(frame, verbose=False)[0]

    def run_midas():
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map[0] = prediction.cpu().numpy()

    t1 = threading.Thread(target=run_yolo)
    t2 = threading.Thread(target=run_midas)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    elapsed = time.time() - start
    return results[0], depth_map[0], elapsed


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 ikkala rejim
    results_seq, depth_seq, time_seq = process_sequential(frame)
    results_par, depth_par, time_par = process_parallel(frame)

    results = results_par
    depth_map = depth_par

    now = time.time()

    # =========================
    # OBJECT PROCESSING
    # =========================
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        depth_value = depth_map[cy, cx]

        # 🔥 MASOFA (approx)
        distance = 1000 / (depth_value + 1e-6)

        # 🔥 chiroyli format
        if distance < 10:
            dist_text = f"{distance:.1f}m"
        else:
            dist_text = f"{int(distance)}m"

        # =========================
        # XAVF DARAJASI
        # =========================
        if distance < 2:
            color = (0, 0, 255)
            status = "Juda yaqin"

            # 🔊 VOICE ALERT (cooldown bilan)
            if label not in last_spoken or now - last_spoken[label] > COOLDOWN:
                speak(f"Diqqat! {label}, {dist_text} masofada")
                last_spoken[label] = now

        elif distance < 5:
            color = (0, 255, 255)
            status = "O‘rtacha"
        else:
            color = (0, 255, 0)
            status = "Uzoq"

        # =========================
        # DRAW
        # =========================
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"{label} {dist_text}",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, status,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # =========================
    # PERFORMANCE INFO
    # =========================
    speedup = time_seq / time_par if time_par > 0 else 0

    cv2.putText(frame, f"Sequential: {round(time_seq,3)} s", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)

    cv2.putText(frame, f"Parallel: {round(time_par,3)} s", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    cv2.putText(frame, f"Speedup: {round(speedup,2)}x", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    cv2.imshow("AI Blind Assist System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# 6

#
# import cv2
# import torch
# import numpy as np
# import pyttsx3
# import threading
# import time
# from ultralytics import YOLO
#
# # -------------------------
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
#
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run).start()
#
# # -------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Device:", device)
#
# # -------------------------
# model_cpu = YOLO("yolov8n.pt")
# model_gpu = YOLO("yolov8n.pt").to(device)
#
# midas_cpu = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas_gpu = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
#
# midas_cpu.eval()
# midas_gpu.to(device).eval()
#
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# # -------------------------
# cap = cv2.VideoCapture(0)
#
# last_spoken = {}
# COOLDOWN = 3
#
# # =========================
# # CPU VERSION
# # =========================
# def process_cpu(frame):
#     start = time.time()
#
#     results = model_cpu(frame, verbose=False)[0]
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img)
#
#     with torch.no_grad():
#         prediction = midas_cpu(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#
#     depth_map = prediction.cpu().numpy()
#
#     return results, depth_map, time.time() - start
#
#
# # =========================
# # GPU VERSION (CUDA)
# # =========================
# def process_gpu(frame):
#     start = time.time()
#
#     results = model_gpu(frame, verbose=False)[0]
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img).to(device)
#
#     with torch.no_grad():
#         prediction = midas_gpu(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()
#
#     depth_map = prediction.cpu().numpy()
#
#     return results, depth_map, time.time() - start
#
#
# # =========================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 🔥 CPU va GPU ishlatish
#     results_cpu, depth_cpu, time_cpu = process_cpu(frame)
#     results_gpu, depth_gpu, time_gpu = process_gpu(frame)
#
#     # 🔥 GPU natijani ko‘rsatamiz
#     results = results_gpu
#     depth_map = depth_gpu
#
#     now = time.time()
#
#     # =========================
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model_gpu.names[cls]
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         depth_value = depth_map[cy, cx]
#
#         distance = 1000 / (depth_value + 1e-6)
#         dist_text = f"{distance:.1f}m"
#
#         # -------------------------
#         if distance < 2:
#             color = (0, 0, 255)
#
#             if label not in last_spoken or now - last_spoken[label] > COOLDOWN:
#                 speak(f"Diqqat! {label}, {dist_text}")
#                 last_spoken[label] = now
#         elif distance < 5:
#             color = (0, 255, 255)
#         else:
#             color = (0, 255, 0)
#
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, f"{label} {dist_text}",
#                     (x1, y1 - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     # =========================
#     # ⏱ TAQQOSLASH NATIJASI
#     speedup = time_cpu / time_gpu if time_gpu > 0 else 0
#
#     cv2.putText(frame, f"CPU Time: {round(time_cpu,3)} s", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
#
#     cv2.putText(frame, f"GPU Time: {round(time_gpu,3)} s", (20,70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#
#     cv2.putText(frame, f"Speedup: {round(speedup,2)}x", (20,100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
#
#     cv2.imshow("CPU vs CUDA Comparison", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



# 7

# import cv2
# import torch
# import numpy as np
# import pyttsx3
# import threading
# import time
# from ultralytics import YOLO
#
# # =========================
# # DEVICE (CPU yoki GPU)
# # =========================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Device:", device)
#
# if device == "cuda":
#     print("GPU:", torch.cuda.get_device_name(0))
#     torch.backends.cudnn.benchmark = True
#
# # =========================
# # 🔊 TEXT TO SPEECH
# # =========================
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)
#
# def speak(text):
#     def run():
#         engine.say(text)
#         engine.runAndWait()
#     threading.Thread(target=run, daemon=True).start()
#
# # =========================
# # 🔥 SPEECH CONTROL
# # =========================
# last_spoken = {}
# COOLDOWN = 3  # sekund
#
# # =========================
# # MODEL LOAD (GPU bilan)
# # =========================
# model = YOLO("yolov8n.pt")
# model.to(device)
# model.fuse()  # tezlashtirish
#
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
# midas.to(device)
# midas.eval()
#
# transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#
# # =========================
# # CAMERA
# # =========================
# cap = cv2.VideoCapture(0)
#
# # =========================
# # SEQUENTIAL
# # =========================
# def process_sequential(frame):
#     start = time.time()
#
#     results = model(frame, device=device, verbose=False)[0]
#
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_batch = transform(img).to(device)
#
#     with torch.no_grad():
#         prediction = midas(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         )
#
#     depth_map = prediction.squeeze().cpu().numpy()
#
#     elapsed = time.time() - start
#     return results, depth_map, elapsed
#
# # =========================
# # PARALLEL
# # =========================
# def process_parallel(frame):
#     start = time.time()
#
#     results = [None]
#     depth_map = [None]
#
#     def run_yolo():
#         results[0] = model(frame, device=device, verbose=False)[0]
#
#     def run_midas():
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         input_batch = transform(img).to(device)
#
#         with torch.no_grad():
#             prediction = midas(input_batch)
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=img.shape[:2],
#                 mode="bicubic",
#                 align_corners=False,
#             )
#
#         depth_map[0] = prediction.squeeze().cpu().numpy()
#
#     t1 = threading.Thread(target=run_yolo)
#     t2 = threading.Thread(target=run_midas)
#
#     t1.start()
#     t2.start()
#
#     t1.join()
#     t2.join()
#
#     elapsed = time.time() - start
#     return results[0], depth_map[0], elapsed
#
# # =========================
# # MAIN LOOP
# # =========================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # ikkala rejimni ishlatish
#     results_seq, depth_seq, time_seq = process_sequential(frame)
#     results_par, depth_par, time_par = process_parallel(frame)
#
#     # parallel natijani ishlatamiz
#     results = results_par
#     depth_map = depth_par
#
#     now = time.time()
#
#     # =========================
#     # OBJECT PROCESSING
#     # =========================
#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#
#         cx = (x1 + x2) // 2
#         cy = (y1 + y2) // 2
#
#         depth_value = depth_map[cy, cx]
#
#         # MASOFA
#         distance = 1000 / (depth_value + 1e-6)
#
#         if distance < 10:
#             dist_text = f"{distance:.1f}m"
#         else:
#             dist_text = f"{int(distance)}m"
#
#         # =========================
#         # XAVF DARAJASI
#         # =========================
#         if distance < 2:
#             color = (0, 0, 255)
#             status = "Juda yaqin"
#
#             if label not in last_spoken or now - last_spoken[label] > COOLDOWN:
#                 speak(f"Diqqat! {label}, {dist_text} masofada")
#                 last_spoken[label] = now
#
#         elif distance < 5:
#             color = (0, 255, 255)
#             status = "O‘rtacha"
#         else:
#             color = (0, 255, 0)
#             status = "Uzoq"
#
#         # =========================
#         # DRAW
#         # =========================
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#
#         cv2.putText(frame, f"{label} {dist_text}",
#                     (x1, y1 - 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#         cv2.putText(frame, status,
#                     (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#     # =========================
#     # PERFORMANCE INFO
#     # =========================
#     speedup = time_seq / time_par if time_par > 0 else 0
#
#     cv2.putText(frame, f"Sequential: {round(time_seq,3)} s", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
#
#     cv2.putText(frame, f"Parallel: {round(time_par,3)} s", (20,70),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
#
#     cv2.putText(frame, f"Speedup: {round(speedup,2)}x", (20,100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
#
#     cv2.imshow("AI Blind Assist System (GPU)", frame)
#
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()







# """
# AI Blind Assist System
# ======================
# Real-time hazard and obstacle detection for visually impaired users.
#
# Pipeline (as described in the article):
#   * YOLOv8   - object detection
#   * MiDaS    - monocular *relative* depth estimation
#   * distance = 1000 / depth  (empirical, NOT metric - see article, Limitations)
#   * three-tier risk classification:
#         high   < 2 m   -> red box,    voice warning (pyttsx3)
#         medium 2-5 m   -> yellow box, visual warning
#         safe   > 5 m   -> green box,  monitoring only
#   * cooldown-controlled voice alerts (no repeated warnings for the same object)
#   * task-level parallelism: YOLOv8 and MiDaS run as two concurrent processes
#     (multiprocessing); voice output runs in a separate thread (threading).
#
# """
#
# import argparse
# import multiprocessing as mp
# import queue
# import threading
# import time
#
# import cv2
# import numpy as np
#
# # --------------------------------------------------------------------------- #
# # Configuration
# # --------------------------------------------------------------------------- #
# HIGH_RISK_M = 2.0          # < 2 m  -> high risk
# MEDIUM_RISK_M = 5.0        # 2-5 m  -> medium risk, > 5 m -> safe
# DISTANCE_K = 1000.0        # distance = DISTANCE_K / depth (empirical formula)
# COOLDOWN_S = 5.0           # min. seconds between voice alerts for same object
# COLORS = {                 # BGR
#     "high": (0, 0, 255),
#     "medium": (0, 255, 255),
#     "safe": (0, 255, 0),
# }
#
#
# # --------------------------------------------------------------------------- #
# # Risk logic
# # --------------------------------------------------------------------------- #
# def classify_risk(distance_m: float) -> str:
#     """Three-tier, distance-based risk classification."""
#     if distance_m < HIGH_RISK_M:
#         return "high"
#     if distance_m <= MEDIUM_RISK_M:
#         return "medium"
#     return "safe"
#
#
# def estimate_distance(depth: np.ndarray, box) -> float:
#     """Approximate distance from the MiDaS value at the bounding-box centre.
#
#     MiDaS returns inverse relative depth (larger value = closer), so the
#     empirical formula distance = 1000 / depth is used. A small patch around
#     the centre is median-filtered to reduce noise.
#     """
#     x1, y1, x2, y2 = box
#     h, w = depth.shape[:2]
#     cx = int(np.clip((x1 + x2) // 2, 0, w - 1))
#     cy = int(np.clip((y1 + y2) // 2, 0, h - 1))
#     r = 2
#     patch = depth[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
#     value = float(np.median(patch))
#     if value <= 1e-6:
#         return float("inf")
#     return DISTANCE_K / value
#
#
# # --------------------------------------------------------------------------- #
# # Models
# # --------------------------------------------------------------------------- #
# class YoloDetector:
#     """YOLOv8 object detector. Returns plain tuples (picklable)."""
#
#     def __init__(self, weights: str = "yolov8n.pt"):
#         import torch
#         from ultralytics import YOLO
#
#         self.device = 0 if torch.cuda.is_available() else "cpu"
#         self.model = YOLO(weights)
#         self(np.zeros((480, 640, 3), np.uint8))          # warm-up
#
#     def __call__(self, frame):
#         result = self.model(frame, verbose=False, device=self.device)[0]
#         detections = []
#         for b in result.boxes:
#             x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
#             detections.append((x1, y1, x2, y2, float(b.conf[0]),
#                                result.names[int(b.cls[0])]))
#         return detections
#
#
# class MidasEstimator:
#     """MiDaS (small) monocular relative depth estimator."""
#
#     def __init__(self):
#         import torch
#
#         self.torch = torch
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
#         self.model.to(self.device).eval()
#         self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
#         self(np.zeros((480, 640, 3), np.uint8))          # warm-up
#
#     def __call__(self, frame) -> np.ndarray:
#         torch = self.torch
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         batch = self.transform(rgb).to(self.device)
#         with torch.no_grad():
#             pred = self.model(batch)
#             pred = torch.nn.functional.interpolate(
#                 pred.unsqueeze(1), size=rgb.shape[:2],
#                 mode="bicubic", align_corners=False).squeeze()
#         return pred.cpu().numpy().astype(np.float32)
#
#
# # --------------------------------------------------------------------------- #
# # Worker processes (one execution unit per model)
# # --------------------------------------------------------------------------- #
# def yolo_worker(frame_q, result_q, ready, weights):
#     detector = YoloDetector(weights)
#     ready.set()
#     while True:
#         item = frame_q.get()
#         if item is None:
#             break
#         frame_id, frame = item
#         t0 = time.perf_counter()
#         detections = detector(frame)
#         result_q.put((frame_id, detections, time.perf_counter() - t0))
#
#
# def midas_worker(frame_q, result_q, ready):
#     estimator = MidasEstimator()
#     ready.set()
#     while True:
#         item = frame_q.get()
#         if item is None:
#             break
#         frame_id, frame = item
#         t0 = time.perf_counter()
#         depth = estimator(frame)
#         result_q.put((frame_id, depth, time.perf_counter() - t0))
#
#
# # --------------------------------------------------------------------------- #
# # Voice alerts (separate thread, cooldown-controlled)
# # --------------------------------------------------------------------------- #
# class VoiceAlert:
#     def __init__(self, cooldown: float = COOLDOWN_S):
#         self.cooldown = cooldown
#         self.last_alert = {}
#         self.messages = queue.Queue(maxsize=5)
#         self.thread = threading.Thread(target=self._run, daemon=True)
#         self.thread.start()
#
#     def _run(self):
#         import pyttsx3
#         engine = pyttsx3.init()          # created inside the thread that uses it
#         while True:
#             message = self.messages.get()
#             if message is None:
#                 break
#             engine.say(message)
#             engine.runAndWait()
#
#     def alert(self, label: str, distance_m: float):
#         now = time.time()
#         if now - self.last_alert.get(label, 0.0) < self.cooldown:
#             return                       # cooldown: suppress repeated alert
#         try:
#             self.messages.put_nowait(
#                 f"Warning! {label} ahead, about {distance_m:.1f} meters")
#             self.last_alert[label] = now
#         except queue.Full:
#             pass
#
#     def close(self):
#         self.messages.put(None)
#
#
# # --------------------------------------------------------------------------- #
# # Frame processing / drawing
# # --------------------------------------------------------------------------- #
# def process_frame(frame, detections, depth, voice=None):
#     """Combine detections + depth, classify risk, draw boxes, trigger alerts."""
#     for x1, y1, x2, y2, _conf, label in detections:
#         distance = estimate_distance(depth, (x1, y1, x2, y2))
#         risk = classify_risk(distance)
#         color = COLORS[risk]
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         text = f"{label} {distance:.1f}m [{risk}]"
#         cv2.putText(frame, text, (x1, max(y1 - 6, 12)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         if risk == "high" and voice is not None:
#             voice.alert(label, distance)
#     return frame
#
#
# class Stats:
#     def __init__(self):
#         self.t_yolo, self.t_midas, self.latency = [], [], []
#         self.t_start = self.t_end = None
#
#     def add(self, t_yolo, t_midas, latency):
#         self.t_yolo.append(t_yolo)
#         self.t_midas.append(t_midas)
#         self.latency.append(latency)
#
#     @property
#     def n(self):
#         return len(self.latency)
#
#     def summary(self):
#         elapsed = self.t_end - self.t_start
#         return {
#             "frames": self.n,
#             "yolo_s": float(np.mean(self.t_yolo)),
#             "midas_s": float(np.mean(self.t_midas)),
#             "total_s": float(np.mean(self.latency)),   # per-frame end-to-end
#             "fps": self.n / elapsed,                    # sustained throughput
#         }
#
#
# def open_source(source: str):
#     cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video source: {source}")
#     return cap
#
#
# def show(frame, fps_text, window="AI Blind Assist System"):
#     cv2.putText(frame, fps_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7, (255, 255, 255), 2)
#     cv2.imshow(window, frame)
#     return cv2.waitKey(1) & 0xFF != ord("q")
#
#
# # --------------------------------------------------------------------------- #
# # Sequential pipeline (baseline)
# # --------------------------------------------------------------------------- #
# def run_sequential(args, voice=None, display=True):
#     detector = YoloDetector(args.weights)
#     estimator = MidasEstimator()
#     cap = open_source(args.source)
#     stats = Stats()
#     stats.t_start = time.perf_counter()
#     try:
#         while args.frames == 0 or stats.n < args.frames:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             t0 = time.perf_counter()
#             detections = detector(frame)
#             t1 = time.perf_counter()
#             depth = estimator(frame)
#             t2 = time.perf_counter()
#             stats.add(t1 - t0, t2 - t1, t2 - t0)
#             frame = process_frame(frame, detections, depth, voice)
#             if display:
#                 fps = stats.n / (time.perf_counter() - stats.t_start)
#                 if not show(frame, f"Sequential  FPS: {fps:.1f}"):
#                     break
#     finally:
#         stats.t_end = time.perf_counter()
#         cap.release()
#         if display:
#             cv2.destroyAllWindows()
#     return stats
#
#
# # --------------------------------------------------------------------------- #
# # Parallel pipeline: YOLOv8 || MiDaS (two processes), main process = capture,
# # synchronisation, risk classification, drawing.
# # --------------------------------------------------------------------------- #
# def run_parallel(args, voice=None, display=True):
#     ctx = mp.get_context("spawn")
#     yolo_in, midas_in = ctx.Queue(maxsize=args.inflight), ctx.Queue(maxsize=args.inflight)
#     yolo_out, midas_out = ctx.Queue(), ctx.Queue()
#     ready = [ctx.Event(), ctx.Event()]
#     procs = [
#         ctx.Process(target=yolo_worker, args=(yolo_in, yolo_out, ready[0], args.weights), daemon=True),
#         ctx.Process(target=midas_worker, args=(midas_in, midas_out, ready[1]), daemon=True),
#     ]
#     for p in procs:
#         p.start()
#     for ev in ready:                       # wait for model load + warm-up
#         ev.wait()
#
#     cap = open_source(args.source)
#     stats = Stats()
#     pending = {}                           # frame_id -> state
#     next_submit = next_done = 0
#     eof = False
#     stats.t_start = time.perf_counter()
#     try:
#         while True:
#             # 1) submit new frames (up to `inflight` frames in the pipeline)
#             while (not eof and len(pending) < args.inflight
#                    and (args.frames == 0 or next_submit < args.frames)):
#                 ok, frame = cap.read()
#                 if not ok:
#                     eof = True
#                     break
#                 pending[next_submit] = {"frame": frame, "t0": time.perf_counter()}
#                 yolo_in.put((next_submit, frame))
#                 midas_in.put((next_submit, frame))
#                 next_submit += 1
#             if args.frames and next_submit >= args.frames:
#                 eof = True
#             if not pending and eof:
#                 break
#
#             # 2) collect results from both workers
#             got = False
#             for q, key, tkey in ((yolo_out, "det", "t_yolo"), (midas_out, "depth", "t_midas")):
#                 try:
#                     while True:
#                         frame_id, result, dt = q.get_nowait()
#                         pending[frame_id][key] = result
#                         pending[frame_id][tkey] = dt
#                         got = True
#                 except queue.Empty:
#                     pass
#             if not got:
#                 time.sleep(0.001)
#
#             # 3) finish frames in order once both results are available
#             while next_done in pending and "det" in pending[next_done] and "depth" in pending[next_done]:
#                 s = pending.pop(next_done)
#                 stats.add(s["t_yolo"], s["t_midas"], time.perf_counter() - s["t0"])
#                 frame = process_frame(s["frame"], s["det"], s["depth"], voice)
#                 next_done += 1
#                 if display:
#                     fps = stats.n / (time.perf_counter() - stats.t_start)
#                     if not show(frame, f"Parallel  FPS: {fps:.1f}"):
#                         eof, pending = True, {}
#                         break
#     finally:
#         stats.t_end = time.perf_counter()
#         for q in (yolo_in, midas_in):
#             try:
#                 q.put_nowait(None)
#             except queue.Full:
#                 pass
#         for p in procs:
#             p.join(timeout=2)
#             if p.is_alive():
#                 p.terminate()
#         cap.release()
#         if display:
#             cv2.destroyAllWindows()
#     return stats
#
#
# # --------------------------------------------------------------------------- #
# # Benchmark: reproduces the quantities reported in Table 1 of the article
# # --------------------------------------------------------------------------- #
# def run_benchmark(args):
#     print("Running sequential baseline ...")
#     seq = run_sequential(args, voice=None, display=False).summary()
#     print("Running parallel pipeline ...")
#     par = run_parallel(args, voice=None, display=False).summary()
#
#     print("\nProcessing   YOLO (s)  MiDaS (s)  Total (s)   FPS")
#     for name, s in (("Sequential", seq), ("Parallel", par)):
#         print(f"{name:<12} {s['yolo_s']:.3f}     {s['midas_s']:.3f}      "
#               f"{s['total_s']:.3f}      {s['fps']:.1f}")
#     print(f"\nThroughput speedup  S_FPS  = FPS_par / FPS_seq = {par['fps'] / seq['fps']:.2f}x")
#     print(f"Latency speedup     S_time = T_seq / T_par     = {seq['total_s'] / par['total_s']:.2f}x")
#     print("(Use --inflight 1 to measure pure per-frame latency without pipeline overlap.)")
#
#
# # --------------------------------------------------------------------------- #
# def main():
#     ap = argparse.ArgumentParser(description="AI Blind Assist System")
#     ap.add_argument("--mode", choices=["parallel", "sequential", "benchmark"], default="parallel")
#     ap.add_argument("--source", default="0", help="webcam index or video file path")
#     ap.add_argument("--weights", default="yolov8n.pt")
#     ap.add_argument("--frames", type=int, default=0, help="max frames (0 = until stream ends / 'q')")
#     ap.add_argument("--inflight", type=int, default=2, help="frames simultaneously in the parallel pipeline")
#     ap.add_argument("--cooldown", type=float, default=COOLDOWN_S, help="voice-alert cooldown, seconds")
#     ap.add_argument("--no-voice", action="store_true")
#     ap.add_argument("--no-show", action="store_true")
#     args = ap.parse_args()
#
#     if args.mode == "benchmark":
#         if args.frames == 0:
#             args.frames = 200
#         run_benchmark(args)
#         return
#
#     voice = None if args.no_voice else VoiceAlert(args.cooldown)
#     runner = run_parallel if args.mode == "parallel" else run_sequential
#     try:
#         stats = runner(args, voice=voice, display=not args.no_show)
#     finally:
#         if voice:
#             voice.close()
#     s = stats.summary()
#     print(f"{args.mode}: {s['frames']} frames, {s['fps']:.2f} FPS, "
#           f"avg per-frame time {s['total_s']:.3f} s")
#
#
# if __name__ == "__main__":
#     mp.freeze_support()          # required for Windows executables
#     main()