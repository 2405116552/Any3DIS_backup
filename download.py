from huggingface_hub import snapshot_download

repo_id = "onmyoji-xiao/ScanNetPP"
cache_dir = "./ScanNet_dataset"  # 指定本地保存目录

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset", 
    local_dir=cache_dir,
    local_dir_use_symlinks=False,  # 下载实际文件副本，而非缓存符号链接
    resume_download=True,          # 支持断点续传
)
print(f"数据集已下载到: {cache_dir}")
