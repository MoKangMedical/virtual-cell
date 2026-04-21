# 贡献指南

感谢您对 virtual-cell 项目的关注！我们欢迎任何形式的贡献。

## 如何贡献

### 报告问题

1. 查看 [Issues](https://github.com/MoKangMedical/virtual-cell/issues) 确保问题未被报告
2. 创建新的 Issue，详细描述问题
3. 包含复现步骤、期望行为和实际行为

### 提交代码

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 遵循项目现有的代码风格
- 添加适当的注释和文档
- 确保所有测试通过
- 保持代码简洁和可读性

## 开发环境设置

### 环境要求

- Python 3.9+
- Node.js 16+（如果需要前端开发）
- Docker（推荐）

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/MoKangMedical/virtual-cell.git
cd virtual-cell
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件
```

4. 运行测试
```bash
python -m pytest tests/
```

## 提交规范

### 提交信息格式

```
<类型>(<范围>): <描述>

[可选的正文]

[可选的脚注]
```

### 类型

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码风格调整（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例

```
feat(api): 添加用户认证接口

- 实现JWT认证
- 添加登录、注册接口
- 添加密码重置功能

Closes #123
```

## Pull Request 指南

### 提交前检查

- [ ] 代码符合项目规范
- [ ] 所有测试通过
- [ ] 文档已更新（如果需要）
- [ ] 提交信息符合规范

### PR 描述

请包含以下信息：

1. **更改内容**：简要描述更改
2. **相关 Issue**：如果有相关 Issue，请引用
3. **测试情况**：描述如何测试更改
4. **截图**：如果有 UI 更改，请提供截图

## 行为准则

### 我们的承诺

为了营造一个开放和友好的环境，我们承诺：

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

### 不可接受的行为

- 使用性化的语言或图像
- 人身攻击或侮辱性评论
- 公开或私下骚扰
- 未经许可发布他人的私人信息
- 其他不道德或不专业的行为

## 许可证

参与本项目即表示您同意您的贡献将在 [MIT License](LICENSE) 下发布。

## 联系方式

如有任何问题，请通过以下方式联系我们：

- 项目 Issues：https://github.com/MoKangMedical/virtual-cell/issues
- 邮箱：contact@mokangmedical.com

感谢您的贡献！
