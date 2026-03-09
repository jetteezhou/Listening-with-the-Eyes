# 字体文件说明

本目录包含用于可视化中文文本的字体文件。

## 字体文件

- `SourceHanSansCN-Regular.otf` - 思源黑体（简体中文），用于可视化图像中的中文文本显示

## 字体来源

字体文件来自 Adobe 的 Source Han Sans（思源黑体）项目：
- GitHub: https://github.com/adobe-fonts/source-han-sans
- 许可证: SIL Open Font License 1.1（开源免费）

## 使用说明

可视化代码会自动优先使用本项目 `fonts` 目录中的字体文件，如果找不到则回退到系统字体。

如果遇到中文乱码问题，请确保 `fonts/SourceHanSansCN-Regular.otf` 文件存在。
