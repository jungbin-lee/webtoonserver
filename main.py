from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import httpx
import os
import asyncio
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
COMFYUI_OUTPUT_DIR = r"C:\Users\bartc\Downloads\ComfyUI_windows_portable\ComfyUI\output"  # ComfyUI 기본 출력 경로
# 상수 정의
COMFY_UI_URL = "http://127.0.0.1:8188"
MAX_WAIT_TIME = 300  # 최대 대기 시간 (초)
POLLING_INTERVAL = 0.5  # 폴링 간격 (초)


class LoraConfig(BaseModel):
    name: str
    model_weight: float = 1.0
    clip_weight: float = 1.0


class GenerateRequest(BaseModel):
    prompt: str
    loras: Optional[List[LoraConfig]] = None
    negative_prompt: str = "dfqwefwefef"
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    steps: int = 20
    cfg: float = 8.0


def load_workflow_template():
    try:
        # 현재 스크립트 파일의 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # workflow 파일 경로 생성
        workflow_path = os.path.join(current_dir, "workflow_webtoon.json")

        # 파일 열기
        with open(workflow_path, "r", encoding='utf-8') as f:
            return json.load(f)

    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Workflow template file not found. Looking in: {workflow_path}"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Invalid workflow template file format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading workflow file: {str(e)}"
        )

def modify_workflow(workflow: Dict[str, Any], request: GenerateRequest) -> Dict[str, Any]:
    try:
        # 프롬프트 설정
        workflow["1"]["inputs"]["text"] = request.prompt
        workflow["20"]["inputs"]["text"] = request.negative_prompt

        # 이미지 크기 설정
        workflow["16"]["inputs"]["width"] = request.width
        workflow["16"]["inputs"]["height"] = request.height

        # Sampling 파라미터 설정
        if request.seed:
            workflow["2"]["inputs"]["seed"] = request.seed
        workflow["2"]["inputs"]["steps"] = request.steps
        workflow["2"]["inputs"]["cfg"] = request.cfg

        # LoRA 설정
        if request.loras and len(request.loras) > 0:
            for idx, lora in enumerate(request.loras[:3], 1):
                workflow["26"]["inputs"][f"switch_{idx}"] = "On"
                workflow["26"]["inputs"][f"lora_name_{idx}"] = lora.name
                workflow["26"]["inputs"][f"model_weight_{idx}"] = lora.model_weight
                workflow["26"]["inputs"][f"clip_weight_{idx}"] = lora.clip_weight

            # 남은 슬롯 비활성화
            for idx in range(len(request.loras) + 1, 4):
                workflow["26"]["inputs"][f"switch_{idx}"] = "Off"
                workflow["26"]["inputs"][f"lora_name_{idx}"] = "None"
        else:
            # 모든 LoRA 비활성화
            for idx in range(1, 4):
                workflow["26"]["inputs"][f"switch_{idx}"] = "Off"
                workflow["26"]["inputs"][f"lora_name_{idx}"] = "None"

        return workflow
    except KeyError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid workflow structure: missing key {str(e)}"
        )


async def queue_prompt(workflow: Dict[str, Any]) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{COMFY_UI_URL}/prompt",
                json={"prompt": workflow}
            )
            response.raise_for_status()
            return response.json()["prompt_id"]
    except httpx.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="ComfyUI server request timeout"
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI server error: {str(e)}"
        )


async def wait_for_image(prompt_id: str) -> str:
    start_time = asyncio.get_event_loop().time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            if asyncio.get_event_loop().time() - start_time > MAX_WAIT_TIME:
                raise HTTPException(
                    status_code=504,
                    detail="Image generation timed out"
                )

            try:
                response = await client.get(f"{COMFY_UI_URL}/history/{prompt_id}")
                response.raise_for_status()

                history = response.json()
                if prompt_id in history and "outputs" in history[prompt_id]:
                    outputs = history[prompt_id]["outputs"]
                    if outputs and "7" in outputs:
                        return outputs["7"]["images"][0]["filename"]

                await asyncio.sleep(POLLING_INTERVAL)

            except httpx.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="ComfyUI server request timeout"
                )
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"ComfyUI server error: {str(e)}"
                )


# @app.post("/generate")
# async def generate_image(request: GenerateRequest):
#     try:
#         # ComfyUI 서버 상태 확인
#         async with httpx.AsyncClient(timeout=5.0) as client:
#             try:
#                 await client.get(COMFY_UI_URL)
#             except httpx.HTTPError:
#                 raise HTTPException(
#                     status_code=503,
#                     detail="ComfyUI server is not accessible"
#                 )
#
#         # 워크플로우 처리
#         workflow = load_workflow_template()
#         modified_workflow = modify_workflow(workflow, request)
#
#         # 이미지 생성 프로세스
#         prompt_id = await queue_prompt(modified_workflow)
#         image_filename = await wait_for_image(prompt_id)
#
#         # 결과 이미지 반환
#         image_path = os.path.join("output", image_filename)
#         if not os.path.exists(image_path):
#             raise HTTPException(
#                 status_code=404,
#                 detail="Generated image file not found"
#             )
#
#         return FileResponse(
#             image_path,
#             media_type="image/png",
#             filename=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#         )
#
#     except Exception as e:
#         if isinstance(e, HTTPException):
#             raise e
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )
@app.post("/generate")
async def generate_image(request: GenerateRequest):
    try:
        # ComfyUI 서버 상태 확인
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.get(COMFY_UI_URL)
            except httpx.HTTPError:
                raise HTTPException(
                    status_code=503,
                    detail="ComfyUI server is not accessible"
                )

        # 워크플로우 처리
        workflow = load_workflow_template()
        modified_workflow = modify_workflow(workflow, request)

        # 이미지 생성 프로세스
        prompt_id = await queue_prompt(modified_workflow)
        image_filename = await wait_for_image(prompt_id)

        # 결과 이미지 경로 (ComfyUI output 디렉토리 사용)
        image_path = os.path.join(COMFYUI_OUTPUT_DIR, image_filename)

        # 파일 존재 여부 디버깅
        if not os.path.exists(image_path):
            available_files = os.listdir(COMFYUI_OUTPUT_DIR)
            raise HTTPException(
                status_code=404,
                detail=f"Generated image file not found at {image_path}. Available files: {available_files}"
            )

        return FileResponse(
            image_path,
            media_type="image/png",
            filename=f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(COMFY_UI_URL)
            return {
                "status": "healthy",
                "comfyui_server": "connected"
            }
    except:
        return {
            "status": "healthy",
            "comfyui_server": "disconnected"
        }
async def queue_prompt(workflow: Dict[str, Any]) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{COMFY_UI_URL}/prompt",
                json={"prompt": workflow}
            )
            response.raise_for_status()
            return response.json()["prompt_id"]
    except TimeoutError:  # httpx.TimeoutError 대신 TimeoutError 사용
        raise HTTPException(
            status_code=504,
            detail="ComfyUI server request timeout"
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI server error: {str(e)}"
        )

async def wait_for_image(prompt_id: str) -> str:
    start_time = asyncio.get_event_loop().time()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                if asyncio.get_event_loop().time() - start_time > MAX_WAIT_TIME:
                    raise HTTPException(
                        status_code=504,
                        detail="Image generation timed out"
                    )

                try:
                    response = await client.get(f"{COMFY_UI_URL}/history/{prompt_id}")
                    response.raise_for_status()

                    history = response.json()
                    if prompt_id in history and "outputs" in history[prompt_id]:
                        outputs = history[prompt_id]["outputs"]
                        if outputs and "7" in outputs:
                            return outputs["7"]["images"][0]["filename"]

                    await asyncio.sleep(POLLING_INTERVAL)

                except TimeoutError:  # httpx.TimeoutError 대신 TimeoutError 사용
                    raise HTTPException(
                        status_code=504,
                        detail="ComfyUI server request timeout"
                    )
                except httpx.HTTPError as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"ComfyUI server error: {str(e)}"
                    )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while waiting for image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)