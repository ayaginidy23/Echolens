import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Blueprint, flash, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import shutil
from datetime import datetime
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from utils import preprocess_video, extract_keyframes_and_events, classify_video, generate_descriptions_and_summary, load_i3d_ucf_finetuned
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "supersecretkey")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'webm'}

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=30)

# Email configuration
app.config['EMAIL_HOST'] = 'smtp.gmail.com'
app.config['EMAIL_PORT'] = 587
app.config['EMAIL_HOST_USER'] = os.getenv('EMAIL_HOST_USER')
app.config['EMAIL_HOST_PASSWORD'] = os.getenv('EMAIL_HOST_PASSWORD')
app.config['EMAIL_USE_TLS'] = True
app.config['SMTP_EMAIL'] = os.getenv('SMTP_EMAIL')  # Add SMTP_EMAIL to app.config


# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize SocketIO
socketio = SocketIO(app)

# Initialize Flask-Session
from flask_session import Session
Session(app)

# Blueprint for main routes
main_bp = Blueprint('main', __name__)

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Utility function to get video duration
def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file for duration calculation: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration



def extract_keyframe_image(video_path, output_image_path, frame_number=0):
    """
    Extract a single frame from a video and save it as an image.
    
    Args:
        video_path (str): Path to the video file.
        output_image_path (str): Path to save the extracted frame.
        frame_number (int): Frame number to extract (default is 0, the first frame).
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"extract_keyframe_image: Could not open video file: {video_path}")
        return False
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        logger.error(f"extract_keyframe_image: Could not read frame {frame_number} from video: {video_path}")
        cap.release()
        return False
    
    success = cv2.imwrite(output_image_path, frame)
    cap.release()
    if not success:
        logger.error(f"extract_keyframe_image: Failed to save image: {output_image_path}")
        return False
    
    logger.info(f"extract_keyframe_image: Successfully extracted frame to: {output_image_path}")
    return True

def generate_pdf_report(output_path, predicted_label, confidence, summary, events, video_duration, keyframe_path):
    """
    Generate a PDF report with the video analysis results.
    
    Args:
        output_path (str): Path to save the PDF report.
        predicted_label (str): Predicted event label.
        confidence (float): Confidence percentage (0-100).
        summary (str): Event summary.
        events (list): List of detected events.
        video_duration (float): Duration of the video in seconds.
        keyframe_path (str): Path to a keyframe image to include in the report.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Custom styles
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Title'],
        fontSize=18,
        textColor=colors.HexColor("#2563EB"),  # Blue color matching the UI
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        name='HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=8
    )
    normal_style = styles['Normal']
    normal_style.fontSize = 12
    normal_style.spaceAfter = 6

    # Title
    elements.append(Paragraph("EchoLens Video Analysis Report", title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # General Information
    elements.append(Paragraph("General Information", heading_style))
    current_time = datetime.datetime.now().strftime("%B %d, %Y at %I:%M %p")
    video_info = [
        f"<b>Analyzed on:</b> {current_time}",
        f"<b>Video Duration:</b> {video_duration:.2f} seconds"
    ]
    for info in video_info:
        elements.append(Paragraph(info, normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Analysis Results
    elements.append(Paragraph("Analysis Results", heading_style))
    results = [
        f"<b>Predicted Event:</b> {predicted_label}",
        f"<b>Confidence:</b> {confidence:.2f}%",
        f"<b>Event Summary:</b> {summary}"
    ]
    for result in results:
        elements.append(Paragraph(result, normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Detected Events Table
    elements.append(Paragraph("Detected Events", heading_style))
    if events:
        table_data = [["Event", "Start Time (s)", "End Time (s)", "Start Frame", "End Frame"]]
        for idx, event in enumerate(events, 1):
            table_data.append([
                f"Event {idx}",
                f"{event['start_time']:.2f}",
                f"{event['end_time']:.2f}",
                str(event['start_frame']),
                str(event['end_frame'])
            ])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563EB")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.grey),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No significant events detected.", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Keyframe Image
    if keyframe_path and os.path.exists(keyframe_path):
        elements.append(Paragraph("Representative Keyframe", heading_style))
        try:
            img = Image(keyframe_path, width=4*inch, height=3*inch)
            elements.append(img)
        except Exception as e:
            logger.error(f"generate_pdf_report: Failed to add image to PDF: {str(e)}")
            elements.append(Paragraph("Failed to include keyframe image.", normal_style))
    else:
        elements.append(Paragraph("No keyframe image available.", normal_style))

    # Build the PDF
    doc.build(elements)
    logger.info(f"generate_pdf_report: Successfully generated PDF report at: {output_path}")


# Routes for static pages
@main_bp.route('/')
def index():
    lang = request.args.get('lang', 'en')
    return render_template('index.html', lang=lang)

@main_bp.route('/about')
def about():
    lang = request.args.get('lang', 'en')
    return render_template('about.html', lang=lang)

@main_bp.route('/features')
def features():
    lang = request.args.get('lang', 'en')
    return render_template('features.html', lang=lang)


@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    lang = request.args.get('lang', 'en')
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        if not name or not email or not message:
            error_message = 'جميع الحقول مطلوبة!' if lang == 'ar' else 'All fields are required!'
            flash(error_message, 'error')
            logger.error("Contact form submission failed: Missing required fields (name: %s, email: %s, message: %s)", name, email, message)
            return redirect(url_for('main.contact', lang=lang))

        msg = MIMEMultipart()
        msg['From'] = f"{app.config['EMAIL_HOST_USER']}"  # Use your email as the actual sender
        msg['To'] = app.config['SMTP_EMAIL']  # To your email
        msg['Reply-To'] = email  # Ensure replies go to the user
        msg['Subject'] = f"New Contact Form Submission from {name} ({email})"  # Include user's email in subject
        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        msg.attach(MIMEText(body, 'plain'))

        try:
            logger.info("Attempting to send email from %s to %s (Reply-To: %s)", app.config['EMAIL_HOST_USER'], app.config['SMTP_EMAIL'], email)
            server = smtplib.SMTP(app.config['EMAIL_HOST'], app.config['EMAIL_PORT'])
            server.set_debuglevel(1)  # Enable SMTP debug output
            server.starttls()
            server.login(app.config['EMAIL_HOST_USER'], app.config['EMAIL_HOST_PASSWORD'])
            server.sendmail(app.config['EMAIL_HOST_USER'], app.config['SMTP_EMAIL'], msg.as_string())
            server.quit()
            success_message = 'شكرًا على رسالتك! سنتواصل معك قريبًا.' if lang == 'ar' else 'Thank you for your message! We will get back to you soon.'
            flash(success_message, 'success')
            logger.info("Email sent successfully to %s", app.config['SMTP_EMAIL'])
        except smtplib.SMTPAuthenticationError as e:
            error_message = 'فشل تسجيل الدخول إلى خادم البريد. تحقق من إعدادات البريد.' if lang == 'ar' else 'Failed to authenticate with the mail server. Check email settings.'
            flash(error_message, 'error')
            logger.error("SMTP Authentication Error: %s", str(e))
        except Exception as e:
            error_message = f'خطأ في إرسال الرسالة: {str(e)}' if lang == 'ar' else f'Error sending message: {str(e)}'
            flash(error_message, 'error')
            logger.error("Failed to send email: %s", str(e))

        return redirect(url_for('main.contact', lang=lang))

    return render_template('contact.html', lang=lang)

@main_bp.route('/analysis')
def analysis():
    lang = request.args.get('lang', 'en')
    return render_template('analysis.html', lang=lang)

@main_bp.route('/upload', methods=['POST'])
def upload_video():
    lang = request.args.get('lang', 'en')
    if 'video' not in request.files:
        return jsonify({'error': 'لم يتم توفير ملف فيديو' if lang == 'ar' else 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'لم يتم تحديد ملف' if lang == 'ar' else 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Get video duration
        video_duration = get_video_duration(video_path)
        if video_duration is None:
            logger.error(f"upload_video: Could not calculate duration for video: {video_path}")
            video_duration = 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"upload_video: Could not open video file: {video_path}")
            return jsonify({'error': 'فشل في فتح ملف الفيديو' if lang == 'ar' else 'Failed to open video file'}), 400
        
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            logger.error(f"upload_video: Could not read frames from video: {video_path}")
            return jsonify({'error': 'فشل في قراءة الفريمات من الفيديو' if lang == 'ar' else 'Failed to read frames from video'}), 400
        cap.release()

        output_video_preprocessed = os.path.join(app.config['OUTPUT_FOLDER'], "output_video_preprocessing.mp4")
        output_video_raw = os.path.join(app.config['OUTPUT_FOLDER'], "keyframes_only_output.mp4")
        output_video_annotated = os.path.join(app.config['OUTPUT_FOLDER'], "keyframes_annotated_output.mp4")
        output_video_significant = os.path.join(app.config['OUTPUT_FOLDER'], "significant_keyframes_output.mp4")
        frames_dir = os.path.join(app.config['OUTPUT_FOLDER'], "frames")

        try:
            logger.info("upload_video: Preprocessing video: %s", video_path)
            preprocess_video(video_path, output_video_preprocessed)

            logger.info("upload_video: Extracting keyframes...")
            events = extract_keyframes_and_events(
                output_video_preprocessed,
                output_video_raw,
                output_video_annotated,
                output_video_significant
            )

            # Check if significant keyframes video was generated and has frames
            cap = cv2.VideoCapture(output_video_significant)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if not os.path.exists(output_video_significant) or total_frames == 0:
                logger.warning("upload_video: No significant keyframes video generated or it is empty")
                return jsonify({
                    'predicted_label': 'No Frames',
                    'confidence': 0.0,
                    'summary': 'No significant frames were extracted for analysis. The video may not contain detectable events.',
                    'events': events
                }), 200

            logger.info("upload_video: Classifying video...")
            labels = ["arrest", "Explosion", "Fight", "normal", "roadaccidents", "shooting", "Stealing", "vandalism"]
            model = load_i3d_ucf_finetuned()
            predicted_label, confidence = classify_video(output_video_significant, model, labels)
            logger.info("upload_video: Initial classification - Label: %s, Confidence: %.2f", predicted_label, confidence)

            logger.info("upload_video: Generating descriptions and summary...")
            frame_paths, descriptions, summary, corrected_label, corrected_confidence = generate_descriptions_and_summary(
                video_path=output_video_significant,
                output_dir=frames_dir,
                predicted_label=predicted_label,
                confidence=confidence
            )
            logger.info("upload_video: After description generation - Corrected Label: %s, Corrected Confidence: %.2f", corrected_label, corrected_confidence)

            # Ensure corrected_confidence is between 0 and 1 before converting to percentage
            if corrected_confidence > 1.0:
                logger.warning("upload_video: Corrected confidence %f exceeds 1.0, normalizing...", corrected_confidence)
                corrected_confidence = corrected_confidence / 100.0
            elif corrected_confidence < 0.0:
                logger.warning("upload_video: Corrected confidence %f is negative, setting to 0", corrected_confidence)
                corrected_confidence = 0.0

            confidence_percentage = round(corrected_confidence * 100, 2)
            confidence_percentage = max(0.0, min(100.0, confidence_percentage))
            logger.info("upload_video: Final confidence percentage: %.2f", confidence_percentage)

            # Store paths and results in session
            session['raw_keyframes_path_analysis'] = output_video_raw
            session['significant_keyframes_path_analysis'] = output_video_significant
            session['analysis_results'] = {
                'predicted_label': corrected_label,
                'confidence': confidence_percentage,
                'summary': summary,
                'events': events,
                'video_duration': video_duration
            }
            logger.info("upload_video: Stored raw keyframes path in session: %s", output_video_raw)
            logger.info("upload_video: Stored significant keyframes path in session: %s", output_video_significant)

            response = {
                'predicted_label': corrected_label,
                'confidence': confidence_percentage,
                'summary': summary,
                'events': events
            }
            return jsonify(response)

        except Exception as e:
            logger.error("upload_video: Error during video analysis: %s", str(e))
            return jsonify({'error': f'خطأ أثناء تحليل الفيديو: {str(e)}' if lang == 'ar' else f'Error during video analysis: {str(e)}'}), 500

        finally:
            # Clean up all files except the raw and significant keyframes videos
            paths_to_remove = [video_path, output_video_preprocessed, output_video_annotated]
            for path in paths_to_remove:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info("upload_video: Removed temporary file: %s", path)
                    except Exception as e:
                        logger.warning("upload_video: Failed to remove temporary file %s: %s", path, str(e))
            if os.path.exists(frames_dir):
                try:
                    shutil.rmtree(frames_dir)
                    logger.info("upload_video: Removed temporary directory: %s", frames_dir)
                except Exception as e:
                    logger.warning("upload_video: Failed to remove temporary directory %s: %s", frames_dir, str(e))

    return jsonify({'error': 'نوع الملف غير صالح' if lang == 'ar' else 'Invalid file type'}), 400

@main_bp.route('/download_raw_analysis')
def download_raw_analysis():
    raw_keyframes_path = session.get('raw_keyframes_path_analysis')
    if not raw_keyframes_path or not os.path.exists(raw_keyframes_path):
        logger.error("download_raw_analysis: Raw keyframes file not found: %s", raw_keyframes_path)
        return jsonify({'error': 'Raw keyframes video not found'}), 404

    try:
        response = send_file(
            raw_keyframes_path,
            as_attachment=True,
            download_name='raw_keyframes.mp4',
            mimetype='video/mp4'
        )
        logger.info("download_raw_analysis: Successfully served file: %s", raw_keyframes_path)
        return response
    except Exception as e:
        logger.error("download_raw_analysis: Error serving file: %s", str(e))
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500
    finally:
        # Clean up the raw keyframes file after download
        if raw_keyframes_path and os.path.exists(raw_keyframes_path):
            try:
                os.remove(raw_keyframes_path)
                logger.info("download_raw_analysis: Removed raw keyframes file: %s", raw_keyframes_path)
                # Clear the session variable
                session.pop('raw_keyframes_path_analysis', None)
            except Exception as e:
                logger.warning("download_raw_analysis: Failed to remove raw keyframes file %s: %s", raw_keyframes_path, str(e))

@main_bp.route('/live', methods=['GET', 'POST'])
def live():
    lang = request.args.get('lang', 'en')
    
    if request.method == 'POST':
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        try:
            if 'action' in request.form:
                action = request.form.get('action')
                
                if action == 'start':
                    if is_ajax:
                        return jsonify({'status': 'recording_started'})
                    return render_template('live.html', recording=True, lang=lang)
                
                elif action == 'stop':
                    if is_ajax:
                        return jsonify({'status': 'recording_stopped'})
                    return render_template('live.html', recording=False, lang=lang)
            
            elif 'video' in request.files:
                video_file = request.files['video']
                
                if video_file.filename == '':
                    logger.error("live: No video selected for analysis")
                    return jsonify({'error': 'No video selected'}), 400
                
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # Fixed datetime.now()
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"live_analysis_{timestamp}.webm")
                video_file.save(video_path)
                logger.info("live: Video file saved at: %s", video_path)

                # Get video duration
                video_duration = get_video_duration(video_path)
                if video_duration is None:
                    logger.error(f"live: Could not calculate duration for video: {video_path}")
                    video_duration = 0.0
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error("live: Could not open video file: %s", video_path)
                    return jsonify({'error': 'Failed to open video file'}), 400
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                cap.release()
                logger.info("live: Total frames in video: %d", frame_count)
                if frame_count == 0:
                    logger.warning("live: No frames found in video: %s", video_path)
                    return jsonify({
                        'predicted_label': 'No Frames',
                        'confidence': 0.0,
                        'summary': 'No frames were extracted from the video.',
                        'events': []
                    }), 200
                
                output_video_preprocessed = os.path.join(app.config['OUTPUT_FOLDER'], "live_preprocessed.mp4")
                output_video_raw = os.path.join(app.config['OUTPUT_FOLDER'], "live_keyframes_only.mp4")
                output_video_annotated = os.path.join(app.config['OUTPUT_FOLDER'], "live_keyframes_annotated.mp4")
                output_video_significant = os.path.join(app.config['OUTPUT_FOLDER'], "live_significant_keyframes.mp4")
                frames_dir = os.path.join(app.config['OUTPUT_FOLDER'], "live_frames")
                
                preprocess_video(video_path, output_video_preprocessed)
                logger.info("live: Video preprocessing completed: %s", output_video_preprocessed)
                
                events = extract_keyframes_and_events(
                    output_video_preprocessed,
                    output_video_raw,
                    output_video_annotated,
                    output_video_significant
                )
                logger.info("live: Keyframes extracted")
                
                if not os.path.exists(output_video_significant):
                    logger.warning("live: No significant keyframes video generated")
                    return jsonify({
                        'predicted_label': 'No Frames',
                        'confidence': 0.0,
                        'summary': 'No significant frames were extracted for analysis.',
                        'events': events
                    }), 200
                
                labels = ["arrest", "Explosion", "Fight", "normal", "roadaccidents", "shooting", "Stealing", "vandalism"]
                model = load_i3d_ucf_finetuned()
                predicted_label, confidence = classify_video(output_video_significant, model, labels)
                logger.info("live: Classification completed - Label: %s, Confidence: %.2f", predicted_label, confidence)
                
                frame_paths, descriptions, summary, corrected_label, corrected_confidence = generate_descriptions_and_summary(
                    video_path=output_video_significant,
                    output_dir=frames_dir,
                    predicted_label=predicted_label,
                    confidence=confidence
                )
                logger.info("live: Summary generated - Corrected Label: %s, Corrected Confidence: %.2f", corrected_label, corrected_confidence)

                # Normalize corrected_confidence
                if corrected_confidence > 1.0:
                    logger.warning("live: Corrected confidence %f exceeds 1.0, normalizing...", corrected_confidence)
                    corrected_confidence = corrected_confidence / 100.0
                elif corrected_confidence < 0.0:
                    logger.warning("live: Corrected confidence %f is negative, setting to 0", corrected_confidence)
                    corrected_confidence = 0.0

                confidence_percentage = round(corrected_confidence * 100, 2)
                confidence_percentage = max(0.0, min(100.0, confidence_percentage))
                logger.info("live: Final confidence percentage: %.2f", confidence_percentage)

                # Store paths and results in session
                session['raw_keyframes_path_live'] = output_video_raw
                session['significant_keyframes_path_live'] = output_video_significant
                session['live_results'] = {
                    'predicted_label': corrected_label,
                    'confidence': confidence_percentage,
                    'summary': summary,
                    'events': events,
                    'video_duration': video_duration
                }
                logger.info("live: Stored raw keyframes path in session: %s", output_video_raw)
                logger.info("live: Stored significant keyframes path in session: %s", output_video_significant)
                
                response = {
                    'predicted_label': corrected_label,
                    'confidence': confidence_percentage,
                    'summary': summary,
                    'events': events
                }
                return jsonify(response)

            elif 'rtsp_url' in request.form:
                rtsp_url = request.form.get('rtsp_url')
                if not rtsp_url:
                    logger.error("live: No RTSP URL provided")
                    return jsonify({"error": "RTSP URL is required"}), 400

                cap = cv2.VideoCapture(rtsp_url)
                if not cap.isOpened():
                    logger.error("live: Failed to open RTSP stream: %s", rtsp_url)
                    return jsonify({"error": "Failed to open RTSP stream"}), 400

                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # Fixed datetime.now()
                temp_video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"rtsp_stream_{timestamp}.mp4")
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

                frame_count = 0
                max_frames = 300
                logger.info("live: Starting frame capture from RTSP stream, FPS: %.2f", fps)
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning("live: Failed to read frame %d from RTSP stream", frame_count)
                        break
                    out.write(frame)
                    frame_count += 1
                cap.release()
                out.release()
                logger.info("live: Captured %d frames from RTSP stream", frame_count)

                # Get video duration
                video_duration = get_video_duration(temp_video_path)
                if video_duration is None:
                    logger.error(f"live: Could not calculate duration for RTSP video: {temp_video_path}")
                    video_duration = 0.0

                if frame_count == 0:
                    logger.error("live: No frames captured from RTSP stream")
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
                    return jsonify({"error": "No frames were captured"}), 400

                output_video_preprocessed = os.path.join(app.config['OUTPUT_FOLDER'], "rtsp_preprocessed.mp4")
                output_video_raw = os.path.join(app.config['OUTPUT_FOLDER'], "rtsp_keyframes_only.mp4")
                output_video_annotated = os.path.join(app.config['OUTPUT_FOLDER'], "rtsp_keyframes_annotated.mp4")
                output_video_significant = os.path.join(app.config['OUTPUT_FOLDER'], "rtsp_significant_keyframes.mp4")
                frames_dir = os.path.join(app.config['OUTPUT_FOLDER'], "rtsp_frames")

                preprocess_video(temp_video_path, output_video_preprocessed)
                logger.info("live: RTSP video preprocessing completed: %s", output_video_preprocessed)

                events = extract_keyframes_and_events(
                    output_video_preprocessed,
                    output_video_raw,
                    output_video_annotated,
                    output_video_significant
                )
                logger.info("live: RTSP keyframes extracted")

                if not os.path.exists(output_video_significant):
                    logger.warning("live: No significant keyframes video generated for RTSP")
                    return jsonify({
                        'predicted_label': 'No Frames',
                        'confidence': 0.0,
                        'summary': 'No significant frames were detected for analysis.',
                        'events': events
                    }), 200

                labels = ["arrest", "Explosion", "Fight", "normal", "roadaccidents", "shooting", "Stealing", "vandalism"]
                model = load_i3d_ucf_finetuned()
                predicted_label, confidence = classify_video(output_video_significant, model, labels)
                logger.info("live: RTSP Classification - Label: %s, Confidence: %.2f", predicted_label, confidence)

                frame_paths, descriptions, summary, corrected_label, corrected_confidence = generate_descriptions_and_summary(
                    video_path=output_video_significant,
                    output_dir=frames_dir,
                    predicted_label=predicted_label,
                    confidence=confidence
                )
                logger.info("live: RTSP descriptions and summary generated - Corrected Label: %s, Corrected Confidence: %.2f", corrected_label, corrected_confidence)

                # Convert corrected_confidence to percentage (0-100) and ensure it’s within bounds
                confidence_percentage = round(corrected_confidence * 100, 2)
                confidence_percentage = max(0.0, min(100.0, confidence_percentage))
                logger.info("live: RTSP Final confidence percentage: %.2f", confidence_percentage)

                # Store paths and results in session
                session['raw_keyframes_path_live'] = output_video_raw
                session['significant_keyframes_path_live'] = output_video_significant
                session['live_results'] = {
                    'predicted_label': corrected_label,
                    'confidence': confidence_percentage,
                    'summary': summary,
                    'events': events,
                    'video_duration': video_duration
                }
                logger.info("live: Stored raw keyframes path in session: %s", output_video_raw)
                logger.info("live: Stored significant keyframes path in session: %s", output_video_significant)

                response = {
                    "predicted_label": corrected_label,
                    "confidence": confidence_percentage,
                    "summary": summary,
                    "events": events
                }
                return jsonify(response)

        except Exception as e:
            logger.error("live: Error in live analysis: %s", str(e), exc_info=True)
            if is_ajax or 'video' in request.files or 'rtsp_url' in request.form:
                return jsonify({'error': str(e)}), 500
            error_message = f"خطأ في التحليل: {str(e)}" if lang == 'ar' else f"Analysis error: {str(e)}"
            return render_template('live.html', error=error_message, lang=lang)
        finally:
            # Clean up all files except the raw and significant keyframes videos
            paths_to_remove = [video_path if 'video' in locals() else None,
                              temp_video_path if 'temp_video_path' in locals() else None,
                              output_video_preprocessed if 'output_video_preprocessed' in locals() else None,
                              output_video_annotated if 'output_video_annotated' in locals() else None]
            for path in paths_to_remove:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info("live: Removed temporary file: %s", path)
                    except Exception as e:
                        logger.warning("live: Failed to remove temporary file %s: %s", path, str(e))
            if 'frames_dir' in locals() and os.path.exists(frames_dir):
                try:
                    shutil.rmtree(frames_dir)
                    logger.info("live: Removed temporary directory: %s", frames_dir)
                except Exception as e:
                    logger.warning("live: Failed to remove temporary directory %s: %s", frames_dir, str(e))
    
    return render_template('live.html', lang=lang)

@main_bp.route('/download_raw_live')
def download_raw_live():
    raw_keyframes_path = session.get('raw_keyframes_path_live')
    if not raw_keyframes_path or not os.path.exists(raw_keyframes_path):
        logger.error("download_raw_live: Raw keyframes file not found: %s", raw_keyframes_path)
        return jsonify({'error': 'Raw keyframes video not found'}), 404

    try:
        response = send_file(
            raw_keyframes_path,
            as_attachment=True,
            download_name='raw_keyframes_live.mp4',
            mimetype='video/mp4'
        )
        logger.info("download_raw_live: Successfully served file: %s", raw_keyframes_path)
        return response
    except Exception as e:
        logger.error("download_raw_live: Error serving file: %s", str(e))
        return jsonify({'error': f'Error serving file: {str(e)}'}), 500
    finally:
        # Clean up the raw keyframes file after download
        if raw_keyframes_path and os.path.exists(raw_keyframes_path):
            try:
                os.remove(raw_keyframes_path)
                logger.info("download_raw_live: Removed raw keyframes file: %s", raw_keyframes_path)
                # Clear the session variable
                session.pop('raw_keyframes_path_live', None)
            except Exception as e:
                logger.warning("download_raw_live: Failed to remove raw keyframes file %s: %s", raw_keyframes_path, str(e))

@main_bp.route('/download_report/<source>')
@main_bp.route('/download_report')
def download_report(source=None):
    if source is None:
        source = request.args.get('source')
    if source not in ['analysis', 'live']:
        return jsonify({'error': 'Invalid source'}), 400

    # Determine session keys based on source
    if source == 'analysis':
        results_key = 'analysis_results'
        significant_keyframes_key = 'significant_keyframes_path_analysis'
        pdf_key = 'analysis_pdf_path'
    else:
        results_key = 'live_results'
        significant_keyframes_key = 'significant_keyframes_path_live'
        pdf_key = 'live_pdf_path'

    # Get results from session
    results = session.get(results_key)
    significant_keyframes_path = session.get(significant_keyframes_key)

    if not results:
        logger.error(f"download_report: No analysis results found in session for source {source}")
        return jsonify({'error': 'Analysis results not found'}), 404

    # Clear previous session data
    session.pop(results_key, None)
    session.pop(significant_keyframes_key, None)
    session.pop(pdf_key, None)

    # Extract a keyframe image (optional, skip if path is invalid or extraction fails)
    keyframe_image_path = None
    if significant_keyframes_path and os.path.exists(significant_keyframes_path):
        keyframe_image_path = os.path.join(app.config['OUTPUT_FOLDER'], f"keyframe_{source}.jpg")
        if not extract_keyframe_image(significant_keyframes_path, keyframe_image_path):
            logger.warning(f"download_report: Failed to extract keyframe image for {source}")
            keyframe_image_path = None
    else:
        logger.warning(f"download_report: Significant keyframes path not found or invalid for source {source}: {significant_keyframes_path}")

    # Generate the PDF report
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], f"report_{source}_{timestamp}.pdf")
    try:
        generate_pdf_report(
            output_path=pdf_path,
            predicted_label=results['predicted_label'],
            confidence=results['confidence'],
            summary=results['summary'],
            events=results['events'],
            video_duration=results['video_duration'],
            keyframe_path=keyframe_image_path
        )
        session[f'{source}_pdf_path'] = pdf_path  # Store PDF path in session
    except Exception as e:
        logger.error(f"download_report: Failed to generate PDF report: {str(e)}")
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500
    finally:
        # Clean up the keyframe image
        if keyframe_image_path and os.path.exists(keyframe_image_path):
            try:
                os.remove(keyframe_image_path)
                logger.info(f"download_report: Removed temporary keyframe image: {keyframe_image_path}")
            except Exception as e:
                logger.warning(f"download_report: Failed to remove keyframe image {keyframe_image_path}: {str(e)}")

    # Send the PDF file
    try:
        response = send_file(
            pdf_path,
            as_attachment=True,
            download_name=f'echolens_report_{source}.pdf',
            mimetype='application/pdf'
        )
        logger.info(f"download_report: Successfully served PDF report: {pdf_path}")
        return response
    except Exception as e:
        logger.error(f"download_report: Error serving PDF report: {str(e)}")
        return jsonify({'error': f'Error serving report: {str(e)}'}), 500
    finally:
        # Clean up the PDF file only
        paths_to_remove = [pdf_path]
        if significant_keyframes_path and os.path.exists(significant_keyframes_path):
            paths_to_remove.append(significant_keyframes_path)
        for path in paths_to_remove:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"download_report: Removed file: {path}")
                except Exception as e:
                    logger.warning(f"download_report: Failed to remove file {path}: {str(e)}")
        # Do not clear session variables here


# Register blueprint
app.register_blueprint(main_bp)

# Clean up resources on shutdown
@app.teardown_appcontext
def cleanup(exception=None):
    cv2.destroyAllWindows()
    logger.info("cleanup: Application context torn down, resources released.")

if __name__ == "__main__":
    socketio.run(app, debug=False, use_reloader=False, host="0.0.0.0", port=5000)