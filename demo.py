import argparse

import cv2
import onnxruntime as ort

from src.app.inference import run_camera_demo, run_image_demo


def main():
    parser = argparse.ArgumentParser(description="Face antispoof demo")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--camera", action="store_true", help="Use webcam")
    parser.add_argument(
        "--face-model",
        type=str,
        default="models/face_detection_yunet_2023mar.onnx",
        help="Face detection ONNX model",
    )
    parser.add_argument(
        "--antispoof-model",
        type=str,
        default="models/best_224.onnx",
        help="Antispoof ONNX model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for real face (default: 0.5)",
    )
    args = parser.parse_args()

    face_detector = cv2.FaceDetectorYN.create(
        str(args.face_model), "", (320, 320), 0.6, 0.3, 5000
    )

    session = ort.InferenceSession(str(args.antispoof_model))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    confidence_threshold = args.threshold

    if args.camera:
        run_camera_demo(
            face_detector, session, input_name, output_name, confidence_threshold
        )
    else:
        if not args.image:
            print("Error: Please provide --image or --camera")
            return
        run_image_demo(
            args.image,
            face_detector,
            session,
            input_name,
            output_name,
            confidence_threshold,
        )


if __name__ == "__main__":
    main()
