"""Inference use cases: Face detection and antispoof classification."""

import cv2

from src.infra.preprocess import preprocess_face, softmax


def process_frame(
    frame, face_detector, session, input_name, output_name, confidence_threshold
):
    """Process a single frame for face detection and antispoof classification."""
    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)

    if faces is not None and len(faces) > 0:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            confidence = face[-1]

            if confidence > 0.5 and w >= 60 and h >= 60:
                face_crop = frame[y : y + h, x : x + w]
                if face_crop.size > 0:
                    face_input = preprocess_face(face_crop)
                    logits = session.run([output_name], {input_name: face_input})[0]
                    probs = softmax(logits[0])
                    live_score = float(probs[0])
                    is_real = live_score >= confidence_threshold

                    status = "Real" if is_real else "Spoof"
                    label = f"{status}: {live_score:.2f}"
                    color = (0, 255, 0) if is_real else (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                    )

    return frame


def run_camera_demo(
    face_detector, session, input_name, output_name, confidence_threshold
):
    """Run the face antispoof demo using webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(
            frame, face_detector, session, input_name, output_name, confidence_threshold
        )

        cv2.imshow("Face Antispoof Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image_demo(
    image_path, face_detector, session, input_name, output_name, confidence_threshold
):
    """Run the face antispoof demo on a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    image = process_frame(
        image, face_detector, session, input_name, output_name, confidence_threshold
    )

    cv2.imshow("Face Antispoof Demo", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
