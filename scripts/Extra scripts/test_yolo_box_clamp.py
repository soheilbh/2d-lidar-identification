def clamp_yolo_box(center, width, min_angle=0, max_angle=359):
    half_w = width / 2
    left_edge = center - half_w
    right_edge = center + half_w
    # Clamp right edge
    if right_edge > max_angle:
        delta = right_edge - max_angle
        center = center - (delta/2)
        width = width -  delta
        left_edge = center - width / 2
        right_edge = center + width / 2
    # Clamp left edge
    if left_edge < min_angle:
        delta = min_angle - left_edge
        center = center + delta/2
        width = width - delta
        left_edge = center - width / 2
        right_edge = center + width / 2
    return center, width, left_edge, right_edge

if __name__ == "__main__":
    # Example values (can be changed for testing)
    center = float(input("Enter center angle: "))
    width = float(input("Enter width: "))
    min_angle = float(input("Enter min angle (default 0): ") or 0)
    max_angle = float(input("Enter max angle (default 359): ") or 359)
    c, w, l, r = clamp_yolo_box(center, width, min_angle, max_angle)
    print(f"Clamped center: {c}")
    print(f"Clamped width: {w}")
    print(f"Left edge: {l}")
    print(f"Right edge: {r}") 