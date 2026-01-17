# GitHub 上传指南

## ✅ 已完成

1. ✅ Git 仓库已初始化
2. ✅ 所有文件已添加到 Git
3. ✅ 初始提交已完成
4. ✅ README.md 已更新（包含完整的训练流程）

## 📋 下一步：推送到 GitHub

### 步骤 1: 在 GitHub 上创建新仓库

1. 登录 GitHub
2. 点击右上角的 "+" → "New repository"
3. 仓库名称填写：`WaveMamba_UGBR`
4. 描述填写：`Wavelet-Enhanced Mamba with Uncertainty-Guided Boundary Refinement for Image Segmentation`
5. 选择 Public 或 Private
6. **不要**初始化 README、.gitignore 或 license（我们已经有了）
7. 点击 "Create repository"

### 步骤 2: 添加远程仓库并推送

在终端中执行以下命令（请将 `YOUR_USERNAME` 替换为您的 GitHub 用户名）：

```bash
cd /home/omnisky/PycharmProject/segdino_sdf_wavelet_ugbr

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/WaveMamba_UGBR.git

# 或者使用 SSH（如果您配置了 SSH key）
# git remote add origin git@github.com:YOUR_USERNAME/WaveMamba_UGBR.git

# 重命名分支为 main（GitHub 默认使用 main）
git branch -M main

# 推送到 GitHub
git push -u origin main
```

### 步骤 3: 验证

访问 `https://github.com/YOUR_USERNAME/WaveMamba_UGBR` 确认代码已成功上传。

## 🔧 如果遇到问题

### 问题 1: 认证失败

如果推送时要求输入用户名和密码，您需要：

1. **使用 Personal Access Token**（推荐）：
   - GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - 生成新 token，勾选 `repo` 权限
   - 使用 token 作为密码

2. **或配置 SSH key**：
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # 然后将 ~/.ssh/id_ed25519.pub 的内容添加到 GitHub Settings → SSH and GPG keys
   ```

### 问题 2: 远程仓库已存在内容

如果远程仓库已经有内容（如 README），需要先拉取：

```bash
git pull origin main --allow-unrelated-histories
# 解决可能的冲突后
git push -u origin main
```

### 问题 3: dinov3 目录问题

如果 dinov3 目录显示为子模块，可以：

```bash
# 移除子模块引用
git rm --cached dinov3
rm -rf dinov3/.git
git add dinov3/
git commit -m "Fix: Add dinov3 as regular directory"
git push origin main
```

## 📝 后续更新

以后更新代码时：

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

## 🎉 完成！

上传完成后，您的仓库将包含：

- ✅ 完整的项目代码
- ✅ 详细的 README.md（包含训练流程）
- ✅ 训练脚本和配置文件
- ✅ 依赖列表（requirements.txt）
- ✅ .gitignore 文件

---

**需要帮助？** 请告诉我您的 GitHub 用户名，我可以帮您生成具体的命令。

