# # 1. Import the library
# from inference_sdk import InferenceHTTPClient
#
# # 2. Connect to your workflow
# client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key=os.environ.get("ROBOFLOW_API_KEY")
# )
#
# # 3. Run your workflow on an image
# result = client.run_workflow(
#     workspace_name="pieters-workspace-kugm8",
#     workflow_id="detect-count-and-visualize",
#     images={
#         "image": "YOUR_IMAGE.jpg" # Path to your image file
#     },
#     use_cache=True # Speeds up repeated requests
# )
#
# # 4. Get your results
# print(result)

import cv2
import os
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# Initialize client
api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY is not set")

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Configure video source (webcam)
source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
config = StreamConfig(
    stream_output=["output_image"],  # Get video back with annotations
    data_output=["count_objects","predictions"],      # Get prediction data via datachannel,
    processing_timeout=3600,             # 60 minutes,
    requested_plan="webrtc-gpu-medium",  # Options: webrtc-gpu-small, webrtc-gpu-medium, webrtc-gpu-large
    requested_region="eu"                # Options: us, eu, ap
)

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize",
    workspace="pieters-workspace-kugm8",
    image_input="image",
    config=config
)

# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

# Handle prediction data via datachannel
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

# Run the session (blocks until closed)
session.run()

