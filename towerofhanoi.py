import cv2 # for webcam and image processing
import mediapipe as mp # for hand tracking
import time # for tracking time
import math # for math functions
import random # for random numbers

# game setup
towers = [[3, 2, 1], [], []] # initial state of towers
held_piece = None # piece picked up by hand
cursor_pos = None # position of the cursor (hand)
start = time.time() # start time
end = None # end time
has_won = False # flag for game win
confetti_sky = [] # list to hold confetti pieces

# hand tracker
mp_hands = mp.solutions.hands # load hand tracking model
hand_tracker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) # set up hand detector
draw_utils = mp.solutions.drawing_utils # for drawing hand landmarks

# confetti
class Confetti:
    def __init__(self, x, y):
        self.x = x # x position
        self.y = y # y position
        self.dx = random.uniform(-1, 1) # x velocity
        self.dy = random.uniform(-4, -1) # y velocity
        self.color = ( # random color
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        self.life = random.randint(60, 100) # lifetime of confetti

    def update(self):
        self.x += self.dx # update x position
        self.y += self.dy # update y position
        self.dy += 0.2 # gravity effect
        self.life -= 1 # decrease life

    def is_alive(self):
        return self.life > 0 # check if confetti is still alive

def dist_between(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y) # distance between two points

def fingers_pinch(landmarks):
    return dist_between(landmarks.landmark[4], landmarks.landmark[8]) < 0.06 # check if fingers are pinched

def rod_index_from_x(x, width):
    return min(x // (width // 3), 2) # get tower index from x position

def is_game_won():
    return towers[2] == [3, 2, 1] # check if all disks are moved to last rod

def render_game_overlay(img):
    screen = img.copy() # copy the frame
    height, width = screen.shape[:2] # get height and width

    for idx, tower in enumerate(towers):
        cx = int((idx + 0.5) * width / 3) # center x of each tower
        cv2.line(screen, (cx, 100), (cx, 400), (128, 128, 128), 10) # draw tower line

        for j, disk in enumerate(reversed(tower)):
            disk_w = 30 + disk * 30 # disk width
            y = 400 - (len(tower) - j) * 20 # disk y position
            col = (0, 0, 0) # disk color
            cv2.rectangle(screen, (cx - disk_w - 2, y - 2), (cx + disk_w + 2, y + 17), (255, 255, 255), -1) # white border
            cv2.rectangle(screen, (cx - disk_w, y), (cx + disk_w, y + 15), col, -1) # disk
            cv2.rectangle(screen, (cx - disk_w, y), (cx + disk_w, y + 15), (255, 255, 255), 2) # disk border

    if held_piece is not None and cursor_pos:
        cx, cy = cursor_pos # cursor position
        disk_w = 30 + held_piece * 30 # width of held disk
        col = (0, 0, 0) # disk color
        cv2.rectangle(screen, (cx - disk_w - 2, cy - 12), (cx + disk_w + 2, cy + 7), (255, 255, 255), -1) # white border
        cv2.rectangle(screen, (cx - disk_w, cy - 10), (cx + disk_w, cy + 5), col, -1) # disk
        cv2.rectangle(screen, (cx - disk_w, cy - 10), (cx + disk_w, cy + 5), (255, 255, 255), 2) # disk border

    total_time = int((end if has_won else time.time()) - start) # total time passed
    cv2.putText(screen, f"Time: {total_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2) # show time

    for piece in confetti_sky:
        if piece.is_alive():
            cv2.circle(screen, (int(piece.x), int(piece.y)), 5, piece.color, -1) # draw confetti

    if has_won:
        cv2.putText(screen, "YOU WIN!", (width // 2 - 150, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3) # show win text

    return screen # return updated frame

cam = cv2.VideoCapture(0) # open webcam

while cam.isOpened():
    ret, frame = cam.read() # read frame
    if not ret:
        break

    frame = cv2.flip(frame, 1) # flip frame horizontally
    h, w, _ = frame.shape # get height and width

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert frame to rgb
    detection = hand_tracker.process(rgb_frame) # detect hands

    if detection.multi_hand_landmarks and not has_won:
        hand = detection.multi_hand_landmarks[0] # first hand landmarks
        draw_utils.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS) # draw landmarks

        x_pos = int(hand.landmark[8].x * w) # x position of index finger
        y_pos = int(hand.landmark[8].y * h) # y position of index finger
        rod_idx = rod_index_from_x(x_pos, w) # find tower based on x
        pinched = fingers_pinch(hand) # check if fingers are pinched

        if pinched:
            if held_piece is None and towers[rod_idx]:
                held_piece = towers[rod_idx].pop() # pick up piece
            cursor_pos = (x_pos, y_pos) # update cursor position
        else:
            if held_piece is not None:
                if not towers[rod_idx] or held_piece < towers[rod_idx][-1]:
                    towers[rod_idx].append(held_piece) # place piece on correct tower
                else:
                    for idx in range(3):
                        if not towers[idx] or held_piece < towers[idx][-1]:
                            towers[idx].append(held_piece) # place piece on any valid tower
                            break
                held_piece = None # reset held piece
                cursor_pos = None # reset cursor

        if is_game_won():
            has_won = True
            end = time.time() # mark end time
            for _ in range(200):
                confetti_sky.append(Confetti(random.randint(0, w), random.randint(0, 50))) # create confetti

    if has_won:
        for c in confetti_sky:
            c.update() # update confetti
        confetti_sky = [c for c in confetti_sky if c.is_alive()] # remove dead confetti

    frame = render_game_overlay(frame) # draw game overlay
    cv2.imshow("tower of hanoi", frame) # show frame

    if cv2.waitKey(5) & 0xFF == 27:
        break # exit when escape key is pressed

cam.release() # release camera
cv2.destroyAllWindows() # close windows