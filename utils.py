import google.generativeai as genai
import cv2
import os
import numpy as np
from PIL import Image
import io
import json
import re
from collections import Counter
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

def preprocess_video(input_path, output_path, target_size=(224, 224)):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        normalized_frame = resized_frame / 255.0
        output_frame = (normalized_frame * 255).astype(np.uint8)
        out.write(output_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def extract_keyframes_and_events(
        video_path="output_video_preprocessing.mp4", 
        output_video_raw="keyframes_only_output.mp4", 
        output_video_annotated="keyframes_annotated_output.mp4", 
        output_video_significant="significant_keyframes_output.mp4",
        output_video_significant_annotated="significant_keyframes_annotated_output.mp4"):

    model = YOLO("yolo12n.pt")
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video file.")
        cap.release()
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input Video FPS: {fps}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writers
    out_raw = cv2.VideoWriter(output_video_raw, fourcc, fps, (frame_width, frame_height))
    out_annotated = cv2.VideoWriter(output_video_annotated, fourcc, fps, (frame_width, frame_height))
    out_significant = cv2.VideoWriter(output_video_significant, fourcc, fps, (frame_width, frame_height))
    out_significant_annotated = cv2.VideoWriter(output_video_significant_annotated, fourcc, fps, (frame_width, frame_height))

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_count = 0
    event_active = False
    event_start_frame = None
    no_motion_threshold = 30
    motion_history = []
    saved_frames = 0
    saved_significant_frames = 0
    events = []
    event_peak_score = 0
    peak_frame = None
    frames_per_keyframe = 10
    significant_frames_per_keyframe = 1
    yolo_interval = 2  # YOLO every 2 frames

    def frame_to_time(frame_num, fps):
        return frame_num / fps

    def check_tracking_event(results):
        if not results or not hasattr(results[0], 'boxes'):
            return False, 0
        boxes = results[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return False, 0
        overlap_score = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                x3, y3, x4, y4 = boxes[j]
                if (x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3):
                    overlap_score += 100
        tracking_score = len(boxes) * 10 + overlap_score
        return True, tracking_score if tracking_score > 0 else len(boxes) * 5

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            if event_active:
                end_time = frame_to_time(frame_count, fps)
                print(f"Event ended at frame {frame_count} ({end_time:.2f}s) - Video ended")
                events[-1]["end_frame"] = frame_count
                events[-1]["end_time"] = end_time
                if peak_frame is not None:
                    for _ in range(frames_per_keyframe):
                        out_raw.write(peak_frame[0])
                        out_annotated.write(peak_frame[1])
                    saved_frames += 1
                    for _ in range(significant_frames_per_keyframe):
                        out_significant.write(peak_frame[0])
                        out_significant_annotated.write(peak_frame[1])
                    saved_significant_frames += 1
            event_active = False
            break

        frame_count += 1
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Motion Detection
        diff_gray = cv2.absdiff(prev_gray, current_gray)
        _, thresh_gray = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        non_zero_count = cv2.countNonZero(thresh_gray)
        motion_history.append(non_zero_count)
        if len(motion_history) > 50:
            motion_history.pop(0)
        adaptive_threshold = max(100, np.mean(motion_history) * 2)

        diff_b = cv2.absdiff(prev_frame[:, :, 0], current_frame[:, :, 0])
        diff_g = cv2.absdiff(prev_frame[:, :, 1], current_frame[:, :, 1])
        diff_r = cv2.absdiff(prev_frame[:, :, 2], current_frame[:, :, 2])
        color_diff = cv2.max(cv2.max(diff_b, diff_g), diff_r)
        _, thresh_color = cv2.threshold(color_diff, 30, 255, cv2.THRESH_BINARY)
        color_change = cv2.countNonZero(thresh_color)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        flow_motion = np.mean(magnitude) > 2

        motion_detected = (non_zero_count > adaptive_threshold) or \
                          (color_change > adaptive_threshold) or \
                          flow_motion

        # YOLOv12 + BoT-SORT Tracking
        tracking_event = False
        tracking_score = 0
        annotated_frame = current_frame.copy()
        if frame_count % yolo_interval == 0:
            results = model.track(
                source=current_frame,
                persist=True,
                tracker="botsort.yaml",
                conf=0.3,
                iou=0.5,
                verbose=False
            )
            tracking_event, tracking_score = check_tracking_event(results)
            if results and results[0].boxes:
                annotated_frame = results[0].plot()

        significant_event = motion_detected or tracking_event

        # Handle Events
        if significant_event:
            if not event_active:
                event_active = True
                event_start_frame = frame_count
                start_time = frame_to_time(frame_count, fps)
                print(f"Event started at frame {frame_count} ({start_time:.2f}s)")
                events.append({"start_frame": frame_count, "start_time": start_time})
                event_peak_score = 0
                peak_frame = None

            motion_score = non_zero_count + color_change + (np.mean(magnitude) * 2.0)
            total_score = motion_score + tracking_score

            if total_score > event_peak_score:
                event_peak_score = total_score
                peak_frame = (current_frame, annotated_frame)

        elif event_active and (frame_count - event_start_frame) > no_motion_threshold:
            event_active = False
            end_time = frame_to_time(frame_count, fps)
            print(f"Event ended at frame {frame_count} ({end_time:.2f}s)")
            events[-1]["end_frame"] = frame_count
            events[-1]["end_time"] = end_time
            if peak_frame is not None:
                for _ in range(frames_per_keyframe):
                    out_raw.write(peak_frame[0])
                    out_annotated.write(peak_frame[1])
                saved_frames += 1
                for _ in range(significant_frames_per_keyframe):
                    out_significant.write(peak_frame[0])
                    out_significant_annotated.write(peak_frame[1])
                saved_significant_frames += 1

        prev_gray = current_gray
        prev_frame = current_frame.copy()

    cap.release()
    out_raw.release()
    out_annotated.release()
    out_significant.release()
    out_significant_annotated.release()

    print(f"\nRaw keyframes video saved as: {output_video_raw}")
    print(f"Annotated keyframes video saved as: {output_video_annotated}")
    print(f"Significant keyframes video saved as: {output_video_significant}")
    print(f"Significant annotated keyframes video saved as: {output_video_significant_annotated}")
    print(f"Total frames written to raw/annotated: {saved_frames} (displayed frames: {saved_frames * frames_per_keyframe})")
    print(f"Total frames written to significant: {saved_significant_frames} (displayed frames: {saved_significant_frames * significant_frames_per_keyframe})")
    print(f"Output video duration (raw/annotated): {saved_frames * frames_per_keyframe / fps:.2f} seconds at {fps} FPS")
    print(f"Output video duration (significant): {saved_significant_frames * significant_frames_per_keyframe / fps:.2f} seconds at {fps} FPS")
    print("\nEvent Summary:")
    for i, event in enumerate(events, 1):
        print(f"Event {i}: Start {event['start_time']:.2f}s (Frame {event['start_frame']}), "
              f"End {event['end_time']:.2f}s (Frame {event['end_frame']})")

    return events


def load_i3d_ucf_finetuned(repo_id="Ahmeddawood0001/i3d_ucf_finetuned", filename="i3d_ucf_finetuned.pth"):
    class I3DClassifier(nn.Module):
        def __init__(self, num_classes):
            super(I3DClassifier, self).__init__()
            self.i3d = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
            self.dropout = nn.Dropout(0.3)
            self.i3d.blocks[6].proj = nn.Linear(2048, num_classes)
        def forward(self, x):
            x = self.i3d(x)
            x = self.dropout(x)
            return x
    device = torch.device("cpu")
    model = I3DClassifier(num_classes=8).to(device)
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def extract_frames(video_path, max_frames=32, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    while len(frames) < max_frames:
        frames.append(frames[-1])
    frames = frames[:max_frames]
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = frames.permute(1, 0, 2, 3)
    cap.release()
    return frames

def classify_video(video_path, model, labels):
    frames = extract_frames(video_path)
    frames = frames.unsqueeze(0).to(torch.device("cpu"))
    with torch.no_grad():
        outputs = model(frames)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = labels[predicted_idx]
        confidence = probabilities[0, predicted_idx].item()
        # Log the raw confidence to verify its range
        print(f"Raw confidence from model: {confidence}")
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        print(f"Clamped confidence: {confidence}")
    return predicted_label, confidence  # Fixed return statement to include both values

def generate_descriptions_and_summary(video_path="significant_keyframes_output.mp4", output_dir="frames", predicted_label=None, confidence=None):
    # Configure the generative AI model
    genai.configure(api_key="AIzaSyCvfIk15FUy-YUuC1hLlfOsf9_J4XoLAGw")
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("Model Loaded Successfully")

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return [], {}, f"Video file {video_path} not found.", predicted_label, confidence

    # Create output directory for frames
    os.makedirs(output_dir, exist_ok=True)

    # Open the video and check if it has frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        print(f"Error: No frames found in video {video_path}.")
        return [], {}, f"No frames found in video {video_path}.", predicted_label, confidence

    # Extract frames at a reduced rate (e.g., every 5th frame)
    frame_rate = 5
    step = max(1, total_frames // frame_rate)

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            success = cv2.imwrite(frame_path, frame)
            if success:
                frames.append(frame_path)
            else:
                print(f"Error: Failed to save frame {frame_path}.")

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frames)} frames")

    # If no frames were extracted, return an error
    if not frames:
        print(f"Error: No frames were successfully extracted from {video_path}.")
        return [], {}, f"No frames were extracted from the video.", predicted_label, confidence

    video_prediction = predicted_label
    print(f"Initial Video Prediction: {video_prediction} with confidence {confidence}")

    descriptions = {}

    # Process each frame to generate descriptions
    for frame_path in frames:
        if not os.path.exists(frame_path):
            print(f"Error: Frame file {frame_path} does not exist.")
            continue

        prompt = (
            f"This frame is from a video initially classified as '{video_prediction}'. "
            "Describe the event happening in the image in one sentence."
        )

        try:
            with open(frame_path, "rb") as img_file:
                image_data = io.BytesIO(img_file.read())
                image_data.seek(0)
                image = Image.open(image_data)

            if model:
                response = model.generate_content([prompt, image])
                descriptions[frame_path] = response.text
            else:
                print("Error: Model is not defined.")
                return frames, descriptions, "Model not defined.", predicted_label, confidence

        except Exception as e:
            print(f"Error processing frame {frame_path}: {str(e)}")
            continue

    print("Descriptions Generated Successfully")

    for frame, desc in descriptions.items():
        print(f"{frame}: {desc}")

    # Define the possible event classes
    event_classes = ["arrest", "Explosion", "Fight", "normal", "roadaccidents", "shooting", "Stealing", "vandalism"]

    # Use Gemini to review descriptions and correct the classification
    corrected_label = predicted_label
    corrected_confidence = confidence

    if descriptions:
        # First attempt: Ask Gemini to classify and return JSON
        correction_prompt = (
            "Here are descriptions of frames from a surveillance video:\n"
            + "\n".join(descriptions.values()) +
            "\nThe video was initially classified as '" + str(predicted_label) + "' with a confidence of " + str(confidence) + ".\n"
            "Based on these descriptions, determine the most appropriate event class from the following options: " + ", ".join(event_classes) + ".\n"
            "Provide the corrected event label and a new confidence score (between 0 and 1) for this classification.\n"
            "Return your response as a JSON string in the following format:\n"
            "{\n"
            "  \"corrected_label\": \"<event_class>\",\n"
            "  \"confidence\": <float between 0 and 1>\n"
            "}\n"
            "Ensure the corrected_label is one of the provided event classes and the confidence is a float between 0 and 1."
        )

        max_attempts = 2
        attempt = 0
        gemini_success = False

        while attempt < max_attempts and not gemini_success:
            attempt += 1
            try:
                correction_response = model.generate_content(correction_prompt)
                response_text = correction_response.text.strip()

                # Try to parse as JSON
                correction_data = json.loads(response_text)
                corrected_label = correction_data.get("corrected_label", predicted_label)
                corrected_confidence = correction_data.get("confidence", confidence)

                # Validate the corrected label and confidence
                if corrected_label not in event_classes:
                    print(f"Error: Gemini returned an invalid label '{corrected_label}'. Retrying...")
                    continue

                corrected_confidence = max(0.0, min(1.0, float(corrected_confidence)))
                gemini_success = True
                print(f"Gemini Corrected Label: {corrected_label} with confidence {corrected_confidence}")

            except json.JSONDecodeError as e:
                print(f"Error parsing Gemini response as JSON: {str(e)}")
                print(f"Gemini response: {response_text}")

                # Try to extract label and confidence from text if JSON parsing fails
                label_match = re.search(r'corrected_label["\s:]+(\w+)', response_text, re.IGNORECASE)
                conf_match = re.search(r'confidence["\s:]+([0-1]?\.\d+|[01])', response_text, re.IGNORECASE)

                if label_match and conf_match:
                    corrected_label = label_match.group(1)
                    if corrected_label not in event_classes:
                        print(f"Extracted invalid label '{corrected_label}'. Retrying...")
                        continue
                    corrected_confidence = max(0.0, min(1.0, float(conf_match.group(1))))
                    gemini_success = True
                    print(f"Extracted from text - Gemini Corrected Label: {corrected_label} with confidence {corrected_confidence}")
                else:
                    print("Could not extract label and confidence from text. Retrying...")
                    continue
            except Exception as e:
                print(f"Error correcting classification with Gemini (attempt {attempt}): {str(e)}")
                continue

        # If Gemini fails after all attempts, fall back to keyword analysis
        if not gemini_success:
            print("Gemini failed to provide a valid response. Falling back to keyword analysis.")

            # Keyword mapping for event classes
            keyword_map = {
                "arrest": ["police", "arrest", "emergency response"],
                "Explosion": ["explosion", "blast", "fire"],
                "Fight": ["fight", "struggle", "conflict"],
                "normal": ["walking", "normal", "calm"],
                "roadaccidents": ["crash", "accident", "bus", "injury", "collide"],
                "shooting": ["shooting", "gun", "firearm"],
                "Stealing": ["steal", "theft", "rob"],
                "vandalism": ["vandalism", "damage", "destroy"]
            }

            # Count occurrences of keywords in descriptions
            event_counts = Counter()
            description_text = " ".join(descriptions.values()).lower()

            for event, keywords in keyword_map.items():
                for keyword in keywords:
                    if keyword in description_text:
                        event_counts[event] += description_text.count(keyword)

            if event_counts:
                corrected_label = event_counts.most_common(1)[0][0]
                # Estimate confidence based on keyword frequency
                total_keywords = sum(event_counts.values())
                corrected_confidence = min(1.0, event_counts[corrected_label] / total_keywords * 0.9)  # Scale down to be conservative
                print(f"Keyword Analysis - Corrected Label: {corrected_label} with confidence {corrected_confidence}")
            else:
                print("No matching keywords found. Falling back to original label and confidence.")
                corrected_label = predicted_label
                corrected_confidence = confidence

    # Generate a summary based on the descriptions
    if not descriptions:
        summary_text = "No descriptions were generated due to processing errors."
        print("Warning: No descriptions available for summary.")
    else:
        summary_prompt = (
            "Here are multiple descriptions of frames from a surveillance video:\n"
            + "\n".join(descriptions.values()) +
            "\nBased on these descriptions, provide a concise summary of the overall event in one paragraph."
        )

        try:
            summary_response = model.generate_content(summary_prompt)
            summary_text = summary_response.text
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            summary_text = "Failed to generate summary due to an error."

    print("\n**Final Summary:**\n")
    print(summary_text)

    print(f"Final Corrected Label: {corrected_label}")
    print(f"Final Confidence (before percentage conversion): {corrected_confidence}")

    return frames, descriptions, summary_text, corrected_label, corrected_confidence