import os
from modelscope import snapshot_download

model_id = "stabilityai/stable-diffusion-2-1-base"
local_dir = "./models"

print(f"正在准备从 ModelScope 下载模型: {model_id}")
print(f"模型文件将保存到: {os.path.abspath(local_dir)}")

try:
    model_dir = snapshot_download(
        model_id,
        cache_dir=local_dir,
    )
    print(f"\n✅ 模型下载成功！")
    print(f"模型文件位于: {model_dir}")
except Exception as e:
    print(f"\n❌ 模型下载失败: {e}")