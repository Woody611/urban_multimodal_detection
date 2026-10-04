# Architecture path（全部来自实际源码，非 YAML 名称推断）

```
RGB(B,G,R) ─┐
IR(1ch)    ─┼─> 已合并为单张 5ch [B,G,R,IR,D] uint8      base.py:430 _merge_channels_rgbid
Depth(1ch) ─┘                                             （cv2.merge((b,g,r,ir,depth))）
                    ↓
        Silence L0 (identity)                              yolo11m_sepstem.yaml:46
                    ↓
  ┌─────────────────┼──────────────────┐
  │ SilenceChannel[0,3)  [3,4)  [4,5)  │  conv.py:364  x[..., c_start:c_end, :, :]
  ↓                 ↓                  ↓
Conv 48 k3 s2    Conv 8 k3 s2     Conv 8 k3 s2          yaml L48/50/52 （三 stem **参数独立**）
  └─────────────────┼──────────────────┘
                    ↓
       Concat([2,4,6]) -> 64ch          ★ FIRST_MODALITY_FUSION  yaml L53
                    ↓
       Conv 128 k3 s2   L8-P2/4         （融合后**立即**通道混合；此后无 modality-specific 分支）
                    ↓
  C3k2 256 (L9) -> Conv 256 s2 -> C3k2 512 (L11) -> Conv 512 s2 -> C3k2 512 (L13)
                    -> Conv 1024 s2 -> C3k2 1024 (L15) -> SPPF (L16) -> C2PSA (L17)
                    ↓
       neck: Upsample+Concat -> C3k2 512 (L20) -> ... -> C3k2 256 (L23 = **P3**) , C3k2 512 (L26 = P4), C3k2 1024 (L29 = P5)
                    ↓
       Detect(head.py:38)，输入 ch = [256, 512, 1024]
                    ↓
        forward (head.py:68-74):  x[i] = cat( cv2[i](x[i]) , cv3[i](x[i]) , 1 )
                                   └── **同一输入 x[i]** ──┘
        cv2 (box) = Sequential(Conv(x,64,3), Conv(64,64,3), Conv2d(64, 64,1))       head.py:47-49
        cv3 (cls) = Sequential( Sequential(DWConv(x,x,3),Conv(x,256,1)),
                                Sequential(DWConv(256,256,3),Conv(256,256,1)),
                                Conv2d(256, 12, 1) )                                head.py:53-60
```

## §3 关键节点

| 节点 | file:line | module/class |
|---|---|---|
| 通道合并（唯一 5ch 入口） | `ultralytics/data/base.py:430` | `_merge_channels_rgbid` |
| IR CLAHE | `ultralytics/data/base.py:334-335` | `cv2.createCLAHE(2.0,(8,8))` |
| 通道切分 | `ultralytics/nn/modules/conv.py:364-370` | `SilenceChannel` |
| 三 stem | `configs/yolo11m_sepstem.yaml:48,50,52` | `Conv 48 / 8 / 8` |
| **第一次融合** | `configs/yolo11m_sepstem.yaml:53` | **`Concat([2,4,6])` → 64ch** |
| 分类分支 | `ultralytics/nn/modules/head.py:53-60` | `Detect.cv3` |
| 回归分支 | `ultralytics/nn/modules/head.py:47-49` | `Detect.cv2` |
| 分支共享输入 | `ultralytics/nn/modules/head.py:68-74` | `Detect.forward` |
