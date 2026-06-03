# ComfyUI sub2api Multimodal Node

本插件提供一个合并节点：

`S2A Multimodal`

- `mode = image`: 文本生成图像，或文本+图像编辑图像
- `mode = text`: 文本+图像理解，输出文本
- 输出固定为 `image` 和 `text` 两个端口；当前模式不用的端口会返回空图或空文本

## 1. 安装

将本目录放到：
`ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui-sub2api-multimodal`

然后重启 ComfyUI。

如果缺依赖：
```bash
python -m pip install -r requirements.txt
```

## 2. 节点参数

- `mode`: 选择 `image` 或 `text`
- `base_url`: 你的 sub2api 或 OpenAI-compatible 中转地址
- `api_key`: 你的 sub2api key
- `model`: 可选 `gpt-image-2`、`gpt-5.5`、`gpt-5.3-codex`
- `prompt`: 提示词
- `action`: 图像模式使用，`auto`/`generate`/`edit`
- `size`: 图像模式使用
- `quality`: 图像模式使用
- `timeout_sec`: 超时时间
- `image_1..image_4`: 可选，多图可同时接入；`action=edit` 时至少接入一张图

## 3.常见报错

- `HTTP 401`: key 错误或已失效
- `HTTP 404`: 该端点不支持；插件会自动尝试备选端点
- `No image found in API response`: 你的中转未返回 `b64_json/url/result` 等图像字段，需要核对该模型是否支持图像输出
- 超时：增大服务端超时或减少输入图片数量/分辨率
