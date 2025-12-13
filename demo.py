import argparse
import cv2
import numpy as np
import onnxruntime as ort


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def preprocess_face(face_img, size=224):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    h, w = face_rgb.shape[:2]
    max_dim = max(h, w)
    delta_w = max_dim - w
    delta_h = max_dim - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    if top or bottom or left or right:
        img = cv2.copyMakeBorder(face_rgb, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    else:
        img = face_rgb
    if img.shape[0] != size or img.shape[1] != size:
        interp = cv2.INTER_LANCZOS4 if img.shape[0] < size else cv2.INTER_AREA
        img = cv2.resize(img, (size, size), interpolation=interp)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = (img - mean) / std
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Face antispoof demo")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--camera", action="store_true", help="Use webcam")
    parser.add_argument("--face-model", type=str, default="models/face_detection_yunet_2023mar.onnx", help="Face detection ONNX model")
    parser.add_argument("--antispoof-model", type=str, default="models/best_224.onnx", help="Antispoof ONNX model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for real face (default: 0.5)")
    args = parser.parse_args()

    face_detector = cv2.FaceDetectorYN.create(
        str(args.face_model),
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    session = ort.InferenceSession(str(args.antispoof_model))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    confidence_threshold = args.threshold

    if args.camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            face_detector.setInputSize((w, h))
            _, faces = face_detector.detect(frame)

            if faces is not None and len(faces) > 0:
                face = faces[0]
                x, y, w, h = map(int, face[:4])
                confidence = face[-1]

                if confidence > 0.5:
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        face_input = preprocess_face(face_crop)
                        logits = session.run([output_name], {input_name: face_input})[0]
                        probs = softmax(logits[0])
                        live_score = float(probs[0])
                        print_score = float(probs[1])
                        replay_score = float(probs[2])
                        spoof_score = print_score + replay_score
                        is_real = live_score >= confidence_threshold
                        max_confidence = max(live_score, spoof_score)

                        status = "Real" if is_real else "Spoof"
                        label = f"{status}: {live_score:.2f}"
                        color = (0, 255, 0) if is_real else (0, 0, 255)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Face Antispoof Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        if not args.image:
            print("Error: Please provide --image or --camera")
            return

        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return

        h, w = image.shape[:2]
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(image)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, w, h = map(int, face[:4])
            confidence = face[-1]

            if confidence > 0.5:
                face_crop = image[y:y+h, x:x+w]
                if face_crop.size > 0:
                    face_input = preprocess_face(face_crop)
                    logits = session.run([output_name], {input_name: face_input})[0]
                    probs = softmax(logits[0])
                    live_score = float(probs[0])
                    print_score = float(probs[1])
                    replay_score = float(probs[2])
                    spoof_score = print_score + replay_score
                    is_real = live_score >= confidence_threshold
                    max_confidence = max(live_score, spoof_score)

                    status = "Real" if is_real else "Spoof"
                    label = f"{status}: {live_score:.2f}"
                    color = (0, 255, 0) if is_real else (0, 0, 255)

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Antispoof Demo", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
