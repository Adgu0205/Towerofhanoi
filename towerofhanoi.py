import cv2
import mediapipe as mp
import time
import math
import random

#game setup
towers = [[3, 2, 1], [], []]
held_piece = None
cursor_pos = None
start = time.time()
end = None
has_won = False
confetti_sky = []

#hand tracker
mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw_utils = mp.solutions.drawing_utils

#confetti
class Confetti:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dx = random.uniform(-1, 1)
        self.dy = random.uniform(-4, -1)
        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        self.life = random.randint(60, 100)

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.dy += 0.2
        self.life -= 1

    def is_alive(self):
        return self.life > 0

def dist_between(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def fingers_pinch(landmarks):
    return dist_between(landmarks.landmark[4], landmarks.landmark[8]) < 0.06

def rod_index_from_x(x, width):
    return min(x // (width // 3), 2)

def is_game_won():
    return towers[2] == [3, 2, 1]

def render_game_overlay(img):
    screen = img.copy()
    height, width = screen.shape[:2]

    for idx, tower in enumerate(towers):
        cx = int((idx + 0.5) * width / 3)
        cv2.line(screen, (cx, 100), (cx, 400), (128, 128, 128), 10)

        for j, disk in enumerate(reversed(tower)):
            disk_w = 30 + disk * 30
            y = 400 - (len(tower) - j) * 20  
            col = (0, 0, 0)
            cv2.rectangle(screen, (cx - disk_w - 2, y - 2), (cx + disk_w + 2, y + 17), (255, 255, 255), -1)
            cv2.rectangle(screen, (cx - disk_w, y), (cx + disk_w, y + 15), col, -1)
            cv2.rectangle(screen, (cx - disk_w, y), (cx + disk_w, y + 15), (255, 255, 255), 2)

    if held_piece is not None and cursor_pos:
        cx, cy = cursor_pos
        disk_w = 30 + held_piece * 30
        col = (0, 0, 0)
        cv2.rectangle(screen, (cx - disk_w - 2, cy - 12), (cx + disk_w + 2, cy + 7), (255, 255, 255), -1)
        cv2.rectangle(screen, (cx - disk_w, cy - 10), (cx + disk_w, cy + 5), col, -1)
        cv2.rectangle(screen, (cx - disk_w, cy - 10), (cx + disk_w, cy + 5), (255, 255, 255), 2)

    total_time = int((end if has_won else time.time()) - start)
    cv2.putText(screen, f"Time: {total_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    for piece in confetti_sky:
        if piece.is_alive():
            cv2.circle(screen, (int(piece.x), int(piece.y)), 5, piece.color, -1)

    if has_won:
        cv2.putText(screen, "YOU WIN!", (width // 2 - 150, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)

    return screen

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection = hand_tracker.process(rgb_frame)

    if detection.multi_hand_landmarks and not has_won:
        hand = detection.multi_hand_landmarks[0]
        draw_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        x_pos = int(hand.landmark[8].x * w)
        y_pos = int(hand.landmark[8].y * h)
        rod_idx = rod_index_from_x(x_pos, w)
        pinched = fingers_pinch(hand)

        if pinched:
            if held_piece is None and towers[rod_idx]:
                held_piece = towers[rod_idx].pop()
            cursor_pos = (x_pos, y_pos)
        else:
            if held_piece is not None:
                if not towers[rod_idx] or held_piece < towers[rod_idx][-1]:
                    towers[rod_idx].append(held_piece)
                else:
                    for idx in range(3):
                        if not towers[idx] or held_piece < towers[idx][-1]:
                            towers[idx].append(held_piece)
                            break
                held_piece = None
                cursor_pos = None

        if is_game_won():
            has_won = True
            end = time.time()
            for _ in range(200):
                confetti_sky.append(Confetti(random.randint(0, w), random.randint(0, 50)))

    if has_won:
        for c in confetti_sky:
            c.update()
        confetti_sky = [c for c in confetti_sky if c.is_alive()]

    frame = render_game_overlay(frame)
    cv2.imshow("tower of hanoi", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()