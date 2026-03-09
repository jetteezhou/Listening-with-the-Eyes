from src.eval_engine import EvaluationEngine
from schemas import EvaluationRequest, TaskResponse, TaskStatus
import sys
import os
import uuid
import json
import asyncio
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Add root directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


app = FastAPI(title="EcoG-Bench API")

# Create API router with prefix
api_router = APIRouter(prefix="/embodied_benchmark/api")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount results directory for static access
# Assuming the app is run from the project root
results_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "results"))
os.makedirs(results_dir, exist_ok=True)
app.mount("/embodied_benchmark/results",
          StaticFiles(directory=results_dir), name="results")

# Mount frontend static files (for production deployment)
frontend_dist_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist_dir):
    # 1. Mount assets specifically to avoid path conflicts with API
    assets_dir = os.path.join(frontend_dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/embodied_benchmark/assets",
                  StaticFiles(directory=assets_dir), name="assets")

    # 2. Redirect root to /embodied_benchmark
    @app.get("/")
    async def root():
        return RedirectResponse(url="/embodied_benchmark/")

    # SPA Catch-all moved to end of file to ensure API routes are matched first

# In-memory task store
tasks = {}


def run_evaluation_task(task_id: str, config: dict):
    tasks[task_id]["status"] = "processing"
    tasks[task_id]["progress"] = 0.0

    def status_callback(msg):
        # Update logs
        tasks[task_id]["logs"].append(
            f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        # Naive progress update based on messages (could be better)
        if "Processing [" in msg:
            try:
                # Parse [1/10]
                parts = msg.split("[")[1].split("]")[0].split("/")
                current = int(parts[0])
                total = int(parts[1])
                tasks[task_id]["progress"] = (current / total) * 100
            except:
                pass

    try:
        engine = EvaluationEngine(config, status_callback=status_callback)
        result_path = engine.run()
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100.0
        tasks[task_id]["result_path"] = result_path
        tasks[task_id]["message"] = "Evaluation completed successfully."
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = str(e)


@api_router.post("/evaluate", response_model=TaskResponse)
async def submit_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    # Create output dir for this task
    if request.test_mode:
        output_dir = os.path.join("results", "web_runs", "test_mode", task_id)
    else:
        output_dir = os.path.join("results", "web_runs", "full_mode", task_id)
    os.makedirs(output_dir, exist_ok=True)

    config = request.model_dump()
    config["output_dir"] = output_dir
    # Ensure test_mode is explicitly in config
    config["test_mode"] = request.test_mode
    # Set default coord_order if not provided: "yx" for gemini, "xy" for others
    if "coord_order" not in config or config["coord_order"] is None:
        config["coord_order"] = "yx" if request.model_provider == "gemini" else "xy"

    tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "Task queued.",
        "logs": [],
        "result_path": None,
        "config": config
    }

    background_tasks.add_task(run_evaluation_task, task_id, config)

    return TaskResponse(
        task_id=task_id,
        status="pending",
        message="Evaluation task started."
    )


@api_router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(**tasks[task_id])


@api_router.get("/results/{task_id}")
async def get_task_results(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")

    result_path = task["result_path"]
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


@api_router.get("/leaderboard")
async def get_leaderboard():
    results_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "results"))
    leaderboard = []

    if not os.path.exists(results_dir):
        return []

    # Define paths to scan: root results and web_runs/full_mode
    scan_dirs = [results_dir]
    web_full_mode_path = os.path.join(results_dir, "web_runs", "full_mode")
    if os.path.exists(web_full_mode_path):
        scan_dirs.append(web_full_mode_path)

    for scan_dir in scan_dirs:
        if not os.path.exists(scan_dir):
            continue

        for model_dir_name in os.listdir(scan_dir):
            model_path = os.path.join(scan_dir, model_dir_name)
            # Skip non-directories and the web_runs container folder itself when scanning root results
            if not os.path.isdir(model_path) or model_dir_name == "web_runs":
                continue

            # Try metrics_summary.json first, then metrics.json
            metrics_path = os.path.join(model_path, "metrics_summary.json")
            if not os.path.exists(metrics_path):
                metrics_path = os.path.join(model_path, "metrics.json")

            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Extract overall scores
                    # If metrics_summary.json (nested under 'overall')
                    if "overall" in data and isinstance(data["overall"], dict):
                        overall = data["overall"]
                    else:
                        # If metrics.json (flat structure)
                        overall = data

                    # Get metadata
                    model_name = data.get("model_name", model_dir_name)
                    provider = data.get("model_provider", "VLM")

                    # Get the modification time of the metrics file as the date
                    mtime = os.path.getmtime(metrics_path)
                    date_str = datetime.fromtimestamp(
                        mtime).strftime('%Y-%m-%d')

                    leaderboard.append({
                        "model": model_name,
                        "provider": provider,
                        "acc_eco":  overall.get("acc_eco", 0) * 100,
                        "acc_cls":  overall.get("acc_cls", 0) * 100,
                        "acc_s":    overall.get("acc_s", 0) * 100,
                        "acc_t":    overall.get("acc_t", 0) * 100,
                        "acc_seq":  overall.get("acc_seq", 0) * 100,
                        "date": date_str
                    })
                except Exception as e:
                    logging.error(f"Error parsing {metrics_path}: {e}")

    # Sort by overall score descending
    leaderboard.sort(key=lambda x: x.get("acc_eco", 0), reverse=True)

    # Add rank
    for i, item in enumerate(leaderboard):
        item["rank"] = i + 1

    return leaderboard

# Include API router at the end to ensure all routes are registered
app.include_router(api_router)

# 3. SPA Catch-all: serve index.html for any other GET request under /embodied_benchmark
# This must be defined AFTER API router include to avoid shadowing API requests
if os.path.exists(frontend_dist_dir):
    @app.get("/embodied_benchmark")
    @app.get("/embodied_benchmark/{full_path:path}")
    async def serve_spa(full_path: str = ""):
        # If the path matches a file in dist (e.g. favicon.ico), serve it
        file_path = os.path.join(frontend_dist_dir, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        return FileResponse(os.path.join(frontend_dist_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow external access (for public deployment)
    uvicorn.run(app, host="0.0.0.0", port=6006)
