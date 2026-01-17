# 推送到 GitHub 指南

## 当前状态

- ✅ 所有代码已提交到本地 Git 仓库
- ✅ README 已更新（包含正确的结果：Epoch 68, MAE=0.0168, Fw_beta=0.9221）
- ✅ 远程仓库已配置：`https://github.com/17863216623/WaveMamba_UGBR.git`
- ⚠️ 需要推送到 GitHub

## 推送方法

### 方法 1：使用 Personal Access Token（推荐）

#### 步骤 1：创建 Personal Access Token

1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 填写 Note（例如：`WaveMamba_UGBR`）
4. 选择过期时间（建议选择较长时间）
5. **重要**：勾选 `repo` 权限（这会自动勾选所有子权限）
6. 点击 "Generate token"
7. **立即复制 Token**（只显示一次！）

#### 步骤 2：使用 Token 推送

```bash
cd /home/omnisky/PycharmProject/segdino_sdf_wavelet_ugbr

# 推送（会提示输入用户名和密码）
git push -u origin main
```

**输入提示时**：
- **Username**: `17863216623`
- **Password**: 粘贴您的 **Personal Access Token**（不是GitHub密码！）

### 方法 2：配置 Git Credential Helper（避免每次输入）

```bash
# 配置 credential helper（保存凭据）
git config --global credential.helper store

# 然后推送（第一次会要求输入，之后会自动保存）
git push -u origin main
```

### 方法 3：在 URL 中直接使用 Token（不推荐，但快速）

```bash
# 将 YOUR_TOKEN 替换为实际的 token
git remote set-url origin https://YOUR_TOKEN@github.com/17863216623/WaveMamba_UGBR.git
git push -u origin main
```

### 方法 4：使用 PyCharm 图形界面推送

1. 打开 PyCharm
2. 点击顶部菜单：`VCS` → `Git` → `Push`
3. 如果提示输入凭据：
   - Username: `17863216623`
   - Password: 您的 Personal Access Token
4. 点击 Push

### 方法 5：配置 SSH（如果网络允许）

如果您想使用 SSH，需要先配置 SSH 密钥：

```bash
# 1. 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"
# 按 Enter 使用默认路径，可以设置密码或留空

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 复制公钥内容，添加到 GitHub：
#    Settings → SSH and GPG keys → New SSH key
#    粘贴公钥内容，保存

# 4. 测试连接
ssh -T git@github.com

# 5. 切换远程 URL 到 SSH
git remote set-url origin git@github.com:17863216623/WaveMamba_UGBR.git

# 6. 推送
git push -u origin main
```

## 当前待推送的提交

```
2498e78 Remove segdino_pic.png
bf65df9 Update README with correct best results: Epoch 68, MAE=0.0168, Fw_beta=0.9221
03a7bd5 Add GitHub setup guide
820f853 Initial commit: WaveMamba_UGBR
```

## 验证推送成功

推送成功后，访问以下 URL 确认：
https://github.com/17863216623/WaveMamba_UGBR

您应该能看到：
- ✅ 完整的项目代码
- ✅ 更新后的 README.md（包含正确的结果数据）
- ✅ 所有提交历史

## 常见问题

### Q: 提示 "Permission denied"
A: 确保使用 Personal Access Token 而不是 GitHub 密码

### Q: 提示 "TLS connection error"
A: 可能是网络问题，尝试：
- 检查网络连接
- 使用代理（如果在中国大陆）
- 稍后重试

### Q: Token 在哪里找到？
A: https://github.com/settings/tokens → 点击您创建的 token 名称

### Q: Token 过期了怎么办？
A: 创建新的 token，然后重新推送

## 推荐流程

1. **创建 Personal Access Token**（方法1的步骤1）
2. **配置 credential helper**（方法2）
3. **推送代码**（方法1的步骤2）

这样以后推送就不需要每次都输入 token 了。

---

**需要帮助？** 如果遇到问题，请告诉我具体的错误信息。

