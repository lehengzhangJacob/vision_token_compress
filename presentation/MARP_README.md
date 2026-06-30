# Marp 使用说明

## 已安装

- **Node.js** v20.18.2 → `~/.local/node/bin/`
- **Marp CLI** v4.1.2 → `~/.local/node/bin/marp`

验证：

```bash
export PATH="$HOME/.local/node/bin:$PATH"
marp --version
```

或使用包装脚本：

```bash
bash presentation/marp.sh --version
```

## 导出命令

```bash
cd /home/msj_team/Jacob/nk/presentation

# HTML（服务器上已验证可用）
bash marp.sh video_token_compression_interview.md --html --no-stdin -o out.html

# PDF / PPTX（需要本机图形环境或 headless Chrome）
bash marp.sh video_token_compression_interview.md --pdf --no-stdin -o out.pdf
bash marp.sh video_token_compression_interview.md --pptx --no-stdin -o out.pptx
```

## 服务器限制

当前 Linux 服务器 **无显示器**，Marp 导出 PPTX/PDF 需启动 Firefox/Chrome，会超时失败。

**推荐在本机（有桌面）使用：**

1. VS Code 安装插件 **「Marp for VS Code」**
2. 打开 `video_token_compression_interview.md`
3. 预览后点击导出 PPTX / PDF

或在有桌面的机器上执行 `marp ... --pptx`。

## MD 加 Marp 头（可选）

在文件最顶部加：

```markdown
---
marp: true
theme: default
paginate: true
size: 16:9
---
```
