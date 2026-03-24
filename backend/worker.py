import redis
import subprocess
import os

r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

OUTPUT_DIR = "/app/shared_volume/completed"
MODEL_PATH = "/app/face_detection_yunet_2023mar.onnx"

print("Worker initialized. Listening to queue...")

while True:
    try:
        # Block until an item is in the queue
        _, task_data = r.brpop("image_processing_queue")
        
        # Split the task_id from the file path
        task_id, input_path = task_data.split('|')
        filename = os.path.basename(input_path)
        output_path = f"{OUTPUT_DIR}/enhanced_{filename}"
        
        print(f"[{task_id}] Running C++ Pipeline...")
        
        subprocess.run(
            ["./enhance_app", input_path, output_path, f"-m={MODEL_PATH}"],
            check=True
        )
        
        # Update Redis so the frontend knows it's done
        r.set(f"status:{task_id}", "completed")
        
        # Cleanup original upload to save space
        if os.path.exists(input_path):
            os.remove(input_path)
            
        print(f"[{task_id}] Success.")
        
    except Exception as e:
        print(f"Error processing task: {e}")
        if 'task_id' in locals():
            r.set(f"status:{task_id}", "failed")