import cv2
import numpy as np
import time
import os
import json
import threading
import re
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment variables.")

try:
    if API_KEY:
        genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring Gemini: {e}")

# Constants
MOTION_THRESHOLD = 5000  # Number of pixels that need to change to be considered motion
TIME_THRESHOLD = 5.0     # Seconds of stillness required
CAM_ID = 0               # Default camera
MIN_AI_INTERVAL = 15  # Minimum seconds between API calls

# Steps
STEPS = [
  {
    "id": 1,
    "instruction": "Connect LED anode to Arduino pin 13 using a 220 resistor",
    "expected": "LED with resistor connected to pin 13"
  },
  {
    "id": 2,
    "instruction": "Connect LED cathode to GND",
    "expected": "LED cathode connected to GND"
  }
]

# Colors (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class ArduinoAssistant:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_ID)
        self.prev_frame = None
        self.last_motion_time = time.time()
        self.current_step_idx = 0
        self.state = "MOVING" # MOVING, STEADY, ANALYZING, FEEDBACK
        self.feedback_data = None
        self.analyzing_thread = None
        self.latest_frame = None
        self.last_ai_call = 0  # Timestamp of last AI call
        self.steady_captured = False  # True = this steady event already triggered AI
        self.quota_exhausted = False  # True = daily quota exhausted, stop all API calls
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            # Fallback for some systems where index 0 might be busy or 1 is default
            self.cap = cv2.VideoCapture(1)
        
        if not self.cap.isOpened():
            raise ValueError("Could not open video device")

    def get_motion_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return 100000 # Force moving on first frame

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_score = np.sum(thresh) / 255
        
        self.prev_frame = gray
        return motion_score

    def analyze_image(self, frame, step):
        """Threaded function to call Gemini"""
        try:
            if not API_KEY:
                # Simulate if no key
                time.sleep(2)
                self.feedback_data = {"status": "correct", "confidence": 1.0, "feedback": "Simulated Success (No API Key)"}
                self.state = "FEEDBACK"
                return

            # Encode image to bytes
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                print("Failed to encode image")
                self.feedback_data = {"status": "error", "feedback": "Camera error"}
                self.state = "FEEDBACK"
                return

            # Prepare prompt
            prompt = f"""
            You are an expert electronics instructor checking a student's Arduino circuit.
            
            Current Step Instruction: "{step['instruction']}"
            Expected Visual: "{step['expected']}"
            
            Analyze the provided image. Does the wiring match the instruction?
            Ignore unrelated objects.
            
            Respond with valid JSON ONLY:
            {{
              "status": "correct" | "partial" | "incorrect",
              "confidence": <float 0.0-1.0>,
              "feedback": "<short, clear guidance string>"
            }}
            """
            
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            image_parts = [
                {"mime_type": "image/jpeg", "data": buffer.tobytes()}
            ]
            
            # Single API call - no retries
            try:
                response = model.generate_content([prompt, image_parts[0]])
            except exceptions.ResourceExhausted:
                print("Quota exceeded. Daily limit reached - stopping all API calls.")
                self.quota_exhausted = True  # Lock all future API calls
                self.feedback_data = {
                    "status": "error",
                    "feedback": "Daily quota exhausted. Try tomorrow."
                }
                self.state = "FEEDBACK"
                return
            except Exception as e:
                print(f"API Error: {e}")
                raise

            if not response:
                raise Exception("Failed to get response from AI")
            
            # Parse JSON
            text = response.text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    result = json.loads(json_str)
                    self.feedback_data = result
                except json.JSONDecodeError:
                    print("JSON Decode Error:", text)
                    self.feedback_data = {
                        "status": "partial", 
                        "confidence": 0.0, 
                        "feedback": "Could not understand AI response. Try again."
                    }
            else:
                print("No JSON found in response:", text)
                self.feedback_data = {
                    "status": "error",
                    "feedback": "Invalid AI response format."
                }
                
        except Exception as e:
            print(f"AI Error: {e}")
            self.feedback_data = {
                "status": "error", 
                "confidence": 0.0, 
                "feedback": f"AI Error: {str(e)}"
            }
        
        self.state = "FEEDBACK"

    def draw_ui(self, frame, motion_score):
        h, w = frame.shape[:2]
        
        # 1. Status Bar Background
        cv2.rectangle(frame, (0, 0), (w, 110), BLACK, -1)
        
        # 2. Step Info
        if self.current_step_idx < len(STEPS):
            step = STEPS[self.current_step_idx]
            text = f"Step {step['id']}: {step['instruction']}"
        else:
            text = "All steps completed!"
            
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        # 3. State Indicator & Feedback
        status_color = WHITE
        status_text = self.state
        feedback_text = None  # Persistent feedback shown below status
        feedback_color = WHITE
        
        # Set persistent feedback from last AI response (shown in all states)
        if self.feedback_data:
            status = self.feedback_data.get("status", "unknown")
            feedback = self.feedback_data.get("feedback", "")
            
            if status == "correct":
                feedback_color = GREEN
                feedback_text = "CORRECT: " + feedback
            elif status == "partial":
                feedback_color = YELLOW
                feedback_text = "PARTIAL: " + feedback
            else:
                feedback_color = RED
                feedback_text = "INCORRECT: " + feedback
        
        if self.state == "FEEDBACK":
            if self.feedback_data:
                status_text = "Feedback received"
                status_color = feedback_color
            else:
                status_text = "Processing..."
                
        elif self.state == "ANALYZING":
            status_text = "Thinking..."
            status_color = YELLOW
            
        elif self.state == "MOVING":
            status_text = "Stabilize camera..."
            status_color = WHITE
            
        elif self.state == "STEADY":
            if self.quota_exhausted:
                status_text = "QUOTA EXHAUSTED. Try tomorrow."
                status_color = RED
            elif self.steady_captured:
                status_text = "Done. Move to retry."
                status_color = YELLOW
            elif time.time() - self.last_ai_call < MIN_AI_INTERVAL:
                remaining = int(MIN_AI_INTERVAL - (time.time() - self.last_ai_call))
                status_text = f"Cooldown ({remaining}s)..."
                status_color = YELLOW
            else:
                status_text = "Hold steady..."
                status_color = GREEN

        cv2.putText(frame, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show persistent feedback below status (keeps showing until new feedback arrives)
        if feedback_text:
            cv2.putText(frame, feedback_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 2)

        # 4. Stability Progress Bar (Bottom)
        if self.state in ["MOVING", "STEADY"]:
            time_still = time.time() - self.last_motion_time
            progress = min(time_still / TIME_THRESHOLD, 1.0)
            bar_width = int(w * progress)
            bar_color = GREEN if progress >= 1.0 else YELLOW
            cv2.rectangle(frame, (0, h-20), (bar_width, h), bar_color, -1)
            
        return frame

    def run(self):
        print("Starting Arduino Assistant...")
        print("Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Mirror frame for better UX
            frame = cv2.flip(frame, 1)
            self.latest_frame = frame.copy()
            
            # Motion Detection
            motion = self.get_motion_score(frame)
            
            # State Machine
            now = time.time()
            
            if self.state == "MOVING":
                if motion < MOTION_THRESHOLD:
                    # If stable for threshold time, switch to STEADY/CAPTURE
                    if now - self.last_motion_time > TIME_THRESHOLD:
                        self.state = "STEADY"
                        self.steady_captured = False  # New steady event
                else:
                    self.last_motion_time = now # Reset timer
            
            elif self.state == "STEADY":
                # Check if quota is exhausted - stop everything
                if self.quota_exhausted:
                    pass  # Do nothing, API is locked
                # Invariant: 1 steady event = 1 AI request, never more
                elif self.steady_captured:
                    # Already sent for this steady event, wait for motion
                    pass
                elif time.time() - self.last_ai_call < MIN_AI_INTERVAL:
                    # Cooldown active, wait (don't retry)
                    pass
                else:
                    # Make ONE AI call for this steady event
                    print("Steady detected! Capturing...")
                    self.steady_captured = True  # Mark: this steady event used
                    if self.current_step_idx < len(STEPS):
                        step = STEPS[self.current_step_idx]
                        self.state = "ANALYZING"
                        self.last_ai_call = time.time()
                        self.analyzing_thread = threading.Thread(
                            target=self.analyze_image, 
                            args=(self.latest_frame, step)
                        )
                        self.analyzing_thread.start()
                    else:
                        self.state = "FEEDBACK"
                        self.feedback_data = {"status": "correct", "feedback": "Complete!"}
                
                # If motion detected while in STEADY, reset (but only if not captured yet)
                if motion > MOTION_THRESHOLD and not self.steady_captured:
                    self.state = "MOVING"
                    self.last_motion_time = now
            
            elif self.state == "ANALYZING":
                # Just wait for thread to update state to FEEDBACK
                pass
            
            elif self.state == "FEEDBACK":
                # If correct, wait a bit then move to next step
                # For this prototype, we'll wait for movement to reset or auto-advance
                if self.feedback_data and self.feedback_data.get("status") == "correct":
                     # Delay or require user acknowledgment? 
                     # Instruction says "Auto-advance when verified"
                     # Let's give it a moment to show the success message, then advance
                     cv2.putText(frame, "Next step in 3s...", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
                     
                     if not hasattr(self, 'success_start'):
                         self.success_start = time.time()
                     
                     if time.time() - self.success_start > 3.0:
                         self.current_step_idx += 1
                         self.state = "MOVING"
                         self.feedback_data = None
                         self.last_motion_time = time.time()
                         delattr(self, 'success_start')
                         
                elif self.feedback_data and self.feedback_data.get("status") in ["incorrect", "partial", "error"]:
                    # If incorrect, go back to monitoring when user moves
                    # Keep feedback_data so it stays visible until new feedback arrives
                    if motion > MOTION_THRESHOLD:
                         self.state = "MOVING"
                         self.last_motion_time = time.time()

            # Draw UI
            frame = self.draw_ui(frame, motion)
            
            cv2.imshow('Arduino Assistant', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset state
                self.state = "MOVING"
                self.feedback_data = None
                self.last_motion_time = time.time()
                print("Reset. Ready for next capture.")
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = ArduinoAssistant()
        app.run()
    except Exception as e:
        print(f"Fatal Error: {e}")
