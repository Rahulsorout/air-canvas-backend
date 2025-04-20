import cv2
import numpy as np
import time
import os
import pygame
from datetime import datetime

# Initialize sound
pygame.mixer.init()
click_sound = pygame.mixer.Sound("click.wav")
eraser_mode = False

def play_click():
    click_sound.play()

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
color_index = 0
brush_thickness = 5

undo_stack = []
redo_stack = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255

def save_to_undo():
    global undo_stack
    undo_stack.append(canvas.copy())
    if len(undo_stack) > 10:
        undo_stack.pop(0)

def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 100, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 140, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

prev_x, prev_y = None, None
drawing_mode = 'freehand'
shape_start = None
shape_select_time = None
shape_confirm_time = None

top_buttons = ['clear', 'undo', 'redo', 'save', 'color1', 'color2', 'color3', 'Erase']
side_buttons = ['freehand', 'rectangle', 'circle', 'thickness-', 'thickness+']
button_areas = {}

def calculate_button_positions(frame):
    global button_areas
    button_areas = {}
    h, w = frame.shape[:2]
    top_bar_height = 60
    side_bar_width = 100

    top_button_width = int(w / len(top_buttons))
    for idx, name in enumerate(top_buttons):
        start_x = idx * top_button_width
        end_x = (idx + 1) * top_button_width
        button_areas[name] = ((start_x, 0), (end_x, top_bar_height))

    side_button_height = int((h - top_bar_height) / len(side_buttons))
    for idx, name in enumerate(side_buttons):
        start_y = top_bar_height + idx * side_button_height
        end_y = start_y + side_button_height
        button_areas[name] = ((0, start_y), (side_bar_width, end_y))

hover_start_time = None
hover_button = None
cursor_position = (0, 0)

def draw_buttons(img):
    for name, (start, end) in button_areas.items():
        shadow_offset = 4
        base_color = (220, 227, 243)
        hover_color = (170, 187, 238)
        active_color = (90, 93, 156)
        text_color = (51, 51, 51)

        is_hover = start[0] < cursor_position[0] < end[0] and start[1] < cursor_position[1] < end[1]
        button_color = hover_color if is_hover else base_color

        if name.startswith('color') and int(name[-1]) - 1 == color_index:
            button_color = colors[color_index]
            text_color = (255, 255, 255)

        if name == 'Erase' and eraser_mode:
            button_color = active_color
            text_color = (255, 255, 255)

        if name in ['freehand', 'rectangle', 'circle'] and drawing_mode == name:
            button_color = active_color
            text_color = (255, 255, 255)

        if name == 'undo' and undo_stack:
            button_color = active_color
            text_color = (255, 255, 255)
        if name == 'redo' and redo_stack:
            button_color = active_color
            text_color = (255, 255, 255)

        cv2.rectangle(img, (start[0] + 3, start[1] + 3), (end[0] - 3, end[1] - 3), button_color, -1)

        label = name.replace('color', 'C').replace('thickness-', '-').replace('thickness+', '+')
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = start[0] + (end[0] - start[0] - text_size[0]) // 2
        text_y = start[1] + (end[1] - start[1] + text_size[1]) // 2
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

def save_canvas():
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    filename = datetime.now().strftime("AirCanvas_%Y%m%d_%H%M%S.png")
    path = os.path.join(downloads_folder, filename)
    cv2.imwrite(path, canvas)
    play_click()

def check_hover_and_click(center):
    global hover_start_time, hover_button, color_index, drawing_mode, brush_thickness, eraser_mode
    current_time = time.time()
    for name, (start, end) in button_areas.items():
        if start[0] <= center[0] <= end[0] and start[1] <= center[1] <= end[1]:
            if hover_button != name:
                hover_button = name
                hover_start_time = current_time
            else:
                if current_time - hover_start_time >= 0.8:
                    play_click()
                    if name != 'Erase':
                        eraser_mode = False
                    if name == 'clear':
                        save_to_undo()
                        canvas[:] = 255
                    elif name == 'save':
                        save_canvas()
                    elif name == 'undo' and undo_stack:
                        redo_stack.append(canvas.copy())
                        canvas[:] = undo_stack.pop()
                    elif name == 'redo' and redo_stack:
                        undo_stack.append(canvas.copy())
                        canvas[:] = redo_stack.pop()
                    elif name.startswith('color'):
                        save_to_undo()
                        color_index = int(name[-1]) - 1
                    elif name in ['freehand', 'rectangle', 'circle']:
                        save_to_undo()
                        drawing_mode = name
                    elif name == 'thickness-':
                        save_to_undo()
                        brush_thickness = max(1, brush_thickness - 1)
                    elif name == 'thickness+':
                        save_to_undo()
                        brush_thickness = min(50, brush_thickness + 1)
                    elif name == 'Erase':
                        save_to_undo()
                        eraser_mode = not eraser_mode
                    hover_start_time = current_time + 1
            return
    hover_button = None
    hover_start_time = None

def is_in_button_area(center):
    return any(start[0] <= center[0] <= end[0] and start[1] <= center[1] <= end[1] for start, end in button_areas.values())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display_frame = frame.copy()
    calculate_button_positions(display_frame)
    draw_buttons(display_frame)
    pen_detected = False

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)

        if area > 500:
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = float(w) / h if h != 0 else 0

            if 0.2 < aspect_ratio < 1.5:
                pen_detected = True
                center = (x + w // 2, y + h // 2)

                check_hover_and_click(center)

                if not is_in_button_area(center):
                    cv2.circle(display_frame, center, 20, (0, 255, 255), 2)

                    now = time.time()
                    if drawing_mode == 'freehand':
                        if prev_x is None or prev_y is None:
                            save_to_undo()  # Save when starting new stroke
                        if prev_x is not None and prev_y is not None:
                            if eraser_mode:
                                cv2.line(canvas, (prev_x, prev_y), center, (255, 255, 255), brush_thickness)
                            else:
                                if center == (prev_x, prev_y):
                                    cv2.circle(canvas, center, brush_thickness, colors[color_index], -1)
                                else:
                                    cv2.line(canvas, (prev_x, prev_y), center, colors[color_index], brush_thickness)
                        prev_x, prev_y = center

                    elif drawing_mode in ['rectangle', 'circle']:
                        if shape_start is None:
                            if shape_select_time is None:
                                shape_select_time = now
                            elif now - shape_select_time >= 2:
                                shape_start = center
                                play_click()
                        else:
                            distance = np.hypot(center[0] - shape_start[0], center[1] - shape_start[1])
                            if distance > 10:
                                if shape_confirm_time is None:
                                    shape_confirm_time = now
                                elif now - shape_confirm_time >= 2:
                                    save_to_undo()
                                    if drawing_mode == 'rectangle':
                                        cv2.rectangle(canvas, shape_start, center, colors[color_index], 2)
                                    elif drawing_mode == 'circle':
                                        radius = int(distance)
                                        cv2.circle(canvas, shape_start, radius, colors[color_index], 2)
                                    play_click()
                                    shape_start = None
                                    shape_select_time = None
                                    shape_confirm_time = None
                            else:
                                shape_confirm_time = None
                        if drawing_mode == 'rectangle' and shape_start is not None:
                            cv2.rectangle(display_frame, shape_start, center, colors[color_index], 2)
                        elif drawing_mode == 'circle' and shape_start is not None:
                            radius = int(np.hypot(center[0] - shape_start[0], center[1] - shape_start[1]))
                            cv2.circle(display_frame, shape_start, radius, colors[color_index], 2)

                    prev_x, prev_y = center
                else:
                    shape_start = None
                    shape_select_time = None
                    shape_confirm_time = None
                    prev_x, prev_y = None, None
            else:
                shape_start = None
                shape_select_time = None
                shape_confirm_time = None
                prev_x, prev_y = None, None
        else:
            shape_start = None
            shape_select_time = None
            shape_confirm_time = None
            prev_x, prev_y = None, None
    else:
        shape_start = None
        shape_select_time = None
        shape_confirm_time = None
        prev_x, prev_y = None, None

    if shape_start is None and drawing_mode != 'freehand':
        cv2.putText(display_frame, "Hold to Set Start", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif shape_confirm_time is None and drawing_mode != 'freehand':
        cv2.putText(display_frame, "Stretch Now", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif shape_confirm_time is not None and drawing_mode != 'freehand':
        cv2.putText(display_frame, "Hold to Confirm", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    combined = display_frame.copy()
    mask_canvas = np.any(canvas != [255, 255, 255], axis=-1)
    combined[mask_canvas] = canvas[mask_canvas]

    border_color = (0, 255, 0) if pen_detected else (0, 0, 255)
    cv2.rectangle(combined, (0, 0), (combined.shape[1]-1, combined.shape[0]-1), border_color, 10)

    cv2.imshow("Air Canvas", combined)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        save_to_undo()
        canvas[:] = 255
    elif key == ord('e'):
        eraser_mode = not eraser_mode
    elif key == ord('u'):
        if undo_stack:
            redo_stack.append(canvas.copy())
            canvas[:] = undo_stack.pop()
    elif key == ord('r'):
        if redo_stack:
            undo_stack.append(canvas.copy())
            canvas[:] = redo_stack.pop()
    elif key == ord('s'):
        save_canvas()

cap.release()
cv2.destroyAllWindows()
