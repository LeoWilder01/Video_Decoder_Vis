# Wan Video Latent Playground

当前完成：官方 TAEW2.1 权重检查、decoder 单独提取、四种 latent 干预、交互网页、真实逐层激活查看、完整尺寸 RGB 解码。尚未完成：真实视频 latent 的语义实验、原版 Wan VAE 对照。不要把随机输入测试当作颜色、物体、运动的语义证据。

## 网页实验室

```sh
cd /Users/yijiangliu/mit/26fall/wan-latent-playground
.venv/bin/python server.py
```

打开 <http://127.0.0.1:8765>。仅绑定本机。默认 seed=42、N(0,1) float32、`[16,21,60,104]`；可在界面切换快速实验尺寸 `[16,5,12,20]`，或以 `--small` 启动。前端为本地 HTML/CSS/JavaScript，无 CDN 或远程服务。

- **INPUT LATENT**：16 个 channel 以 4×4 的体积剖面排列。滚轮移动时间截面，点击像素选择坐标；数字表格与单数值栏支持按住上下拖动，Shift 微调，也可直接键入精确数值。支持区域、整 channel、时间段加法与恢复初始值。数值表为局部 7×9 窗口，完整截面使用共用色标。
- **DECODER WEIGHTS**：64 个真实参数 tensor，按实际模块编号排序。卷积核 `[O,I,KH,KW]` 以 O 为盒子、I 为剖面深度，KH/KW 为截面；每页16个输出通道，偏置作为单值查看。全部数值来自官方 checkpoint，没有样本占位。权重只读。
- **FORWARD TRACE**：23 个外层操作，可以展开 MemBlock 和 TGrow 内部。点击一个操作，选择该层 t，实际运行到这个位置并读取完整精度的激活。可从所选 latent 截面开始追踪，逐个操作前进/后退，跳转查看对应权重。时间扩张时下一步默认追踪该位置派生的第一个时间位置，也可手动选择另一位置。
- **RGB OUTPUT**：真实 decoder 生成 MP4；可逐帧查看编码前的浮点 RGB、任意像素的三通道数值及其8位显示值。未裁剪的最后一层为84帧，实际输出裁去最前3帧，得到81帧。

编辑与切片浏览即时响应；完整视频使用按钮重新计算。实测完整尺寸 CPU 解码约25秒（4线程，此机器本次测试），不宣称实时。所有结果带 revision，输入变更后明确标记旧视频和旧激活。计算任务使用不可变的输入副本，后续编辑不会混入正在进行的解码。

为限制内存，只保留一个中间时间截面的全部通道，并按需传输某个通道；不会存储整条网络的全部激活。CPU float32 解码，源权重 FP16 转 float32 不改变参数数值。hook 在原地 ReLU 之前复制输出，确保 Conv 与 ReLU 的观察结果不被混淆。

网页数据在运行进程中保存；关闭服务前可导出当前 `.npy`。生成视频与原始 RGB 存在 `experiments/web/`，当前进程保留最近两次解码结果。完整 float32 RGB 每次约388 MB，MP4仅用于观看，精确值读取 `.npy`。随机视频仅用于数值实验。

回归测试：

```sh
.venv/bin/python -m unittest test_server -v
```

测试覆盖单值修改及版本冲突、三种批量 mask 和边界、流式 RGB 与上游解码逐数一致（含时间裁剪）、卷积 / MemBlock 内部 / 时间扩张的中间值捕获。

## 本地文件

- `weights/taew2_1.safetensors`：官方完整权重，22,642,902 字节；包括 encoder 和 decoder。
- `weights/taew2_1_decoder.safetensors`：无损提取的 decoder，19,695,110 字节。
- `reports/weights-inventory.json`：128 个权重张量的名称、形状、参数数、min/max/mean/std。
- `reports/provenance.json`：官方源地址和固定 commit。
- `vendor/taehv.py` / `vendor/LICENSE`：固定版本的上游实现及其许可证。
- `inspect_weights.py`：标准库即可运行，重新提取 decoder 并检查每个数值。
- `experiment.py`：仅为 decoder 分配模型权重；不加载 encoder、T5 或 DiT。

官方来源：<https://github.com/madebyollin/taehv>，固定 commit `011dfc2112197741c540e0bdd5b7b67bcc930771`。

## 运行

在此目录运行：

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
python3 inspect_weights.py
python3 inspect_weights.py --tensor decoder.1.weight > first-layer.json
.venv/bin/python experiment.py --synthetic --mode scalar --deltas -1 0 1 --out experiments/new-smoke
```

已有 `.venv`，已装依赖。`requirements-lock.txt` 记录本机 Python 3.9/macOS 的实际版本；其他平台可使用 `requirements.txt`。

真实 latent 必须为无 batch 的 `[16,T,H,W]`，以 `.npy` 或 `.safetensors` 保存。safetensors 默认 key 为 `latent`，可通过 `--key` 指定。必须确认数据来自扩散模型最终输出的归一化 latent 空间；原始 VAE posterior 均值需要先转换，不能仅凭形状判断。

```sh
.venv/bin/python experiment.py --latent /absolute/path/latent.npy --latent-space diffusion --mode region --channel 3 --time 8 --y 28 --x 48 --extent 1 4 4 --units channel_std --deltas -2 -1 -0.5 0 0.5 1 2 --device cpu --out experiments/real-region
```

`--mode`：scalar（一个数）、region（单通道小区域）、channel（整个通道）、time（所有通道的一段时间）。`--extent DT DY DX` 用于区域；time 模式只取 DT。所有索引从零开始。每次编辑均从不可变 baseline 重新计算，不累计 slider 改动。`channel_std` 表示按 baseline 每通道标准差缩放；`absolute` 为原始数值增量。`--fps` 应填写源视频帧率。

输出包括 baseline/各扰动 MP4、逐帧 RGB 平均绝对差、每帧差异图数组、选中 latent slice 修改前后数值、解码耗时。差异在 MP4 压缩前、decoder 输出 clamp 后计算。范围统一为 RGB `[0,1]`。输入原始 latent、mask 和 delta 足够重建每次编辑。

## 接口与架构

TAE 接收 `[N,T,C,H,W]`，直接使用 diffusion-space latent；不要再套原版 Wan decoder 的 mean/std 逆变换。其输出为 `[N,4T-3,3,8H,8W]`，RGB `[0,1]`。本例 `[16,21,60,104]` 得到 81 帧 480×832。

简化 decoder：

```
Clamp: 3*tanh(z/3)
Conv2d 16→256, 3×3 + ReLU
3×MemBlock(256)
空间×2，时间×1，Conv2d 256→128
3×MemBlock(128)
空间×2，时间×2，Conv2d 128→64
3×MemBlock(64)
空间×2，时间×2，Conv2d 64→64
ReLU + Conv2d 64→3
RGB clamp [0,1]，裁去开头3帧
```

MemBlock 将当前特征与前一时间位置的输入特征按通道拼接，经过三层卷积与残差连接。时间扩张由 1×1 卷积加 reshape 完成。它不是逐帧完全独立的解码器，也不是原版 Wan 的 3D VAE。

第一层权重 `[256,16,3,3]`：256 个输出 filter，每个 filter `[16,3,3]`；固定输入/输出通道后，看到一个 3×3 数值切片。第一组 MemBlock 内 `[256,512,3,3]` 的 512 为当前256通道加过去256通道。

## 实验顺序

1. 获得一段真实 clean latent 和其 baseline 视频。可以从已有 Wan/FastVideo 采样导出；若先用 TAE encoder 编码普通视频，必须标记为 TAE-encoded 输入，之后再验证真正的生成 latent。
2. 固定 seed/输入/解码参数，先做零扰动对照，再做 ±0.25、±0.5、±1、±2 个通道标准差的对称扫描；小 scalar、区域、通道和时间分别比较。
3. 显示同步 baseline/edited 视频、固定色标 latent heatmap、RGB 差异图、逐帧误差曲线。一个 scalar 可能只造成微弱像素变化，因此不能只看肉眼播放结果。
4. 比较中心与边缘、早期与晚期的干预。不要假设一个 channel 对应固定颜色或物体；语义结论应跨视频、位置、幅度验证。
5. 将关键现象送入原版 Wan VAE 对照。TAE 有自身近似误差、入口 tanh 饱和和输出 clamp。
6. 网页已接入 decoder 常驻服务：baseline 不变，保留修改日志和 revision，忽略过时响应；完整解码显式触发，小尺寸用于快速实验。不要用权重文件大小预测显存或全分辨率 FPS。

## 已验证

- 提取 decoder 与官方完整模型在同一小输入下逐数一致。
- sequential 与 parallel 解码在小输入下数值近似一致。
- 四种编辑 mask 的修改元素数正确。
- 随机 `[16,3,8,8]` 实际解码成 `[1,9,3,64,64]`，零扰动差异为零，±1 scalar 产生非零差异。
- 全尺寸 `[1,21,16,60,104] → [1,81,3,480,832]` 已完成 CPU 实际解码与播放，约25秒；两处 scalar 修改后重新解码，检测到非零 RGB 差异，记录于 `reports/web-validation.json`。

尚无真实 clean latent，因此本阶段没有关于 channel 语义、物体或运动变化的实证结论。

## Geometric sections (current UI)

The original overview, three rails, nested module representations, dimension
labels and bottom summary are preserved. The separate study has been removed.

Only selected latent/output arrays expand, as noninteractive, untextured wireframe
placeholders. At most eight rows follow the face edges. Weight tensors never
expand; c and input-channel i remain inspectable in the existing preview. Only
the representative on the main axis loads and draws actual values. Placeholder
bounds never affect camera limits, and no background all-channel fetch runs.

Every primary latent volume's front top-right corner lies on the same axis.
Parameter links show input → weight → bias → convolution output; bias is omitted
where the actual model has none. Memory blocks retain three separate convolution
parameter groups, connected to their corresponding internal output nodes.

Main intermediate data plates use one linear spatial scale: 120×208 matches
the input 60×104 plate size; 60×104 is half as wide and half as tall. The input
size stays unchanged. Larger stages receive more axial spacing. Small internal
module thumbnails remain schematic. True dimensions appear in the preview. Full grids are never reduced to make
numbers fit. First double-click frames a member with a 260% ceiling, second shows
numeric cells, third returns to the overview. Wheel changes the selected group's
depth; Ctrl/Command + wheel zooms. Pan bounds stay fixed when a placeholder array is expanded.

Activation values require Compute for the desired time slice. Only the inspected channel plane loads;
uncomputed times are empty, never duplicated or invented.

Overview notes are transparent screen-space text and stay visible at every zoom.
Decorative MemBlock brackets, rail drop lines, and operation ticks are removed.
Computation arrows attach to the midpoints of the left/right vertical face edges;
bias placement aligns its input port horizontally with the weight output port.
