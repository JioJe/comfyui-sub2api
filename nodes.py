import base64
import io
import json
import uuid
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image


DEFAULT_TIMEOUT = 90
DEFAULT_BASE_URL = ""


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _request_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTP {e.code} @ {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error @ {url}: {e}") from e


def _request_multipart(
    url: str,
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    api_key: str,
    timeout: int,
) -> dict:
    boundary = f"----ComfySub2Api{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode("utf-8")
    body = io.BytesIO()

    for key, value in fields.items():
        if value is None or value == "":
            continue
        body.write(b"--" + boundary_bytes + b"\r\n")
        body.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.write(str(value).encode("utf-8"))
        body.write(b"\r\n")

    for field_name, filename, payload, content_type in files:
        body.write(b"--" + boundary_bytes + b"\r\n")
        body.write(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.write(payload)
        body.write(b"\r\n")

    body.write(b"--" + boundary_bytes + b"--\r\n")

    req = urllib.request.Request(url, data=body.getvalue(), method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTP {e.code} @ {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error @ {url}: {e}") from e


def _try_json(base_url: str, endpoints: list[str], payload: dict, api_key: str, timeout: int) -> tuple[dict, str]:
    base = _normalize_base_url(base_url)
    errors = []
    for endpoint in endpoints:
        url = f"{base}{endpoint}"
        try:
            return _request_json(url, payload, api_key, timeout), url
        except RuntimeError as e:
            errors.append(str(e))
    raise RuntimeError("; ".join(errors))


def _try_multipart(
    base_url: str,
    endpoints: list[str],
    fields: dict[str, str],
    files: list[tuple[str, str, bytes, str]],
    api_key: str,
    timeout: int,
) -> tuple[dict, str]:
    base = _normalize_base_url(base_url)
    errors = []
    for endpoint in endpoints:
        url = f"{base}{endpoint}"
        try:
            return _request_multipart(url, fields, files, api_key, timeout), url
        except RuntimeError as e:
            errors.append(str(e))
    raise RuntimeError("; ".join(errors))


def _tensor_to_pil(image_tensor: torch.Tensor, max_side: int | None = None) -> Image.Image:
    if image_tensor is None:
        raise ValueError("image tensor is required")
    if image_tensor.ndim != 4 or image_tensor.shape[0] < 1:
        raise ValueError("invalid IMAGE tensor shape")

    arr = image_tensor[0].cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr).convert("RGB")

    if max_side and max_side > 0:
        width, height = image.size
        current_max = max(width, height)
        if current_max > max_side:
            scale = float(max_side) / float(current_max)
            resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(resized, Image.Resampling.LANCZOS)

    return image


def _tensor_to_png_bytes(image_tensor: torch.Tensor, max_side: int | None = None) -> bytes:
    image = _tensor_to_pil(image_tensor, max_side=max_side)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _tensor_to_data_url(image_tensor: torch.Tensor, max_side: int | None = None) -> str:
    raw = _tensor_to_png_bytes(image_tensor, max_side=max_side)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _decode_image_base64(encoded: str) -> Image.Image:
    if "," in encoded and encoded.split(",", 1)[0].startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _images_to_response_content(images: list[torch.Tensor], max_side: int | None = None) -> list[dict]:
    content = []
    for image in images:
        if image is None:
            continue
        content.append({"type": "input_image", "image_url": _tensor_to_data_url(image, max_side=max_side)})
    return content


def _extract_text_from_response(data: dict) -> str:
    if not isinstance(data, dict):
        return ""

    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]

    chunks = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("text"), str):
                chunks.append(item["text"])
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
    if chunks:
        return "\n".join(x for x in chunks if x and x.strip()).strip()

    if isinstance(data.get("text"), str):
        return data["text"]
    return ""


def _extract_text_from_chat_completion(data: dict) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else None
    if not first:
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


def _download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def _pil_to_comfy_image(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return torch.from_numpy(arr)[None, ...]


def _empty_image() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _extract_image_from_payload(data: dict) -> torch.Tensor:
    if not isinstance(data, dict):
        raise RuntimeError("invalid image response payload")

    items = data.get("data")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            encoded = item.get("b64_json") or item.get("base64")
            if isinstance(encoded, str) and encoded.strip():
                return _pil_to_comfy_image(_decode_image_base64(encoded))
            url = item.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return _pil_to_comfy_image(_download_image(url))

    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            encoded = item.get("result") or item.get("b64_json") or item.get("base64") or item.get("image_base64")
            if isinstance(encoded, str) and encoded.strip():
                return _pil_to_comfy_image(_decode_image_base64(encoded))
            url = item.get("url") or item.get("image_url")
            if isinstance(url, str) and url.startswith("http"):
                return _pil_to_comfy_image(_download_image(url))
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                encoded = part.get("result") or part.get("b64_json") or part.get("base64") or part.get("image_base64")
                if isinstance(encoded, str) and encoded.strip():
                    return _pil_to_comfy_image(_decode_image_base64(encoded))
                url = part.get("url") or part.get("image_url")
                if isinstance(url, str) and url.startswith("http"):
                    return _pil_to_comfy_image(_download_image(url))

    for key in ("image", "image_url", "url", "b64_json", "base64", "result", "image_base64"):
        value = data.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return _pil_to_comfy_image(_download_image(value))
        if isinstance(value, str) and value.strip() and key not in ("image_url", "url"):
            return _pil_to_comfy_image(_decode_image_base64(value))

    raise RuntimeError("no image found in API response")


def _run_text_mode(base_url, api_key, model, prompt, timeout_sec, images):
    if not api_key.strip():
        raise RuntimeError("api_key is required")
    timeout = max(15, int(timeout_sec))

    response_content = []
    if prompt.strip():
        response_content.append({"type": "input_text", "text": prompt})
    response_content.extend(_images_to_response_content(images, max_side=1024))

    if not response_content:
        raise RuntimeError("provide prompt and/or at least one image")

    response_errors = []
    try:
        payload = {"model": model, "input": [{"role": "user", "content": response_content}]}
        data, _ = _try_json(base_url, ["/v1/responses", "/responses"], payload, api_key, timeout)
        text = _extract_text_from_response(data)
        return text or json.dumps(data, ensure_ascii=False)
    except Exception as e:
        response_errors.append(f"responses failed: {e}")

    try:
        chat_content = []
        if prompt.strip():
            chat_content.append({"type": "text", "text": prompt})
        for image in images:
            if image is not None:
                chat_content.append({"type": "image_url", "image_url": {"url": _tensor_to_data_url(image, 1024)}})
        payload = {"model": model, "messages": [{"role": "user", "content": chat_content}]}
        data, _ = _try_json(base_url, ["/v1/chat/completions", "/chat/completions"], payload, api_key, timeout)
        text = _extract_text_from_chat_completion(data)
        return text or json.dumps(data, ensure_ascii=False)
    except Exception as e:
        response_errors.append(f"chat/completions failed: {e}")
        raise RuntimeError("; ".join(response_errors))


def _run_image_mode(base_url, api_key, model, prompt, action, size, quality, timeout_sec, images):
    if not api_key.strip():
        raise RuntimeError("api_key is required")

    timeout = max(30, int(timeout_sec))
    max_side = 1024 if size == "auto" else int(size.split("x")[0]) if "x" in size else 1024
    images = [img for img in images if img is not None]
    image_data_urls = [_tensor_to_data_url(img, max_side=max_side) for img in images]
    image_png_bytes = [_tensor_to_png_bytes(img, max_side=max_side) for img in images]

    if action == "edit" and not images:
        raise RuntimeError("edit mode requires at least one input image")

    fields = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "response_format": "b64_json",
    }
    image_errors = []

    if images:
        multipart_file_fields = ["image", "images", "images[]"]
        for field_name in multipart_file_fields:
            files = []
            for index, raw in enumerate(image_png_bytes, start=1):
                files.append((field_name, f"input_{index}.png", raw, "image/png"))
            try:
                data, _ = _try_multipart(
                    base_url,
                    ["/v1/images/edits", "/images/edits"],
                    fields,
                    files,
                    api_key,
                    timeout,
                )
                return _extract_image_from_payload(data)
            except Exception as e:
                image_errors.append(f"images/edits multipart ({field_name}) failed: {e}")

        json_candidates = [
            ("images[].image_url=string", {"images": [{"image_url": url} for url in image_data_urls]}),
            ("images[].image_url.url", {"images": [{"image_url": {"url": url}} for url in image_data_urls]}),
            ("images[]=string", {"images": list(image_data_urls)}),
        ]
        for label, extra_payload in json_candidates:
            try:
                payload = dict(fields)
                payload.update(extra_payload)
                data, _ = _try_json(base_url, ["/v1/images/edits", "/images/edits"], payload, api_key, timeout)
                return _extract_image_from_payload(data)
            except Exception as e:
                image_errors.append(f"images/edits json ({label}) failed: {e}")

        if action == "edit":
            raise RuntimeError("; ".join(image_errors))

    if action != "edit":
        try:
            payload = dict(fields)
            data, _ = _try_json(
                base_url,
                ["/v1/images/generations", "/images/generations"],
                payload,
                api_key,
                timeout,
            )
            return _extract_image_from_payload(data)
        except Exception as e:
            image_errors.append(f"images/generations failed: {e}")

        try:
            content = [{"type": "input_text", "text": prompt}]
            content.extend(_images_to_response_content(images, max_side=max_side))
            payload = {
                "model": model,
                "input": [{"role": "user", "content": content}],
                "modalities": ["image"],
            }
            data, _ = _try_json(base_url, ["/v1/responses", "/responses"], payload, api_key, timeout)
            return _extract_image_from_payload(data)
        except Exception as e:
            image_errors.append(f"responses image fallback failed: {e}")

    model_hint = ""
    if "image" not in (model or "").lower():
        model_hint = " Hint: use an image-capable model such as gpt-image-2 or your provider's image model alias."
    raise RuntimeError("; ".join(image_errors) + model_hint)


class S2AMultimodal:
    CATEGORY = "sub2api"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["image", "text"],),
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["gpt-image-2", "gpt-5.5", "gpt-5.3-codex"],),
                "prompt": ("STRING", {"default": "Describe or edit this image.", "multiline": True}),
                "action": (["auto", "generate", "edit"],),
                "size": (["1024x1024", "1024x1536", "1536x1024", "auto"],),
                "quality": (["low", "medium", "high", "auto"],),
                "timeout_sec": ("INT", {"default": 120, "min": 15, "max": 900, "step": 5}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    def run(
        self,
        mode,
        base_url,
        api_key,
        model,
        prompt,
        action,
        size,
        quality,
        timeout_sec,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        images = [image_1, image_2, image_3, image_4]
        if mode == "text":
            text = _run_text_mode(base_url, api_key, model, prompt, timeout_sec, images)
            return (_empty_image(), text)

        image = _run_image_mode(base_url, api_key, model, prompt, action, size, quality, timeout_sec, images)
        return (image, "")


NODE_CLASS_MAPPINGS = {
    "S2AMultimodal": S2AMultimodal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "S2AMultimodal": "S2A Multimodal",
}
