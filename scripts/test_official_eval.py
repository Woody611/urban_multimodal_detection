"""scripts/test_official_eval.py — 冻结评测器 regression tests（brief §十二 的 Test 1-6）。

只读 + 纯内存断言，不写任何文件、不训练。
用法:  python scripts/test_official_eval.py     (exit 0 = 全过)
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import official_eval as OE  # noqa: E402

FAILS = []


def ck(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def _mk(pred_rows):
    return np.array(pred_rows, np.float32) if pred_rows else np.zeros((0, 6), np.float32)


# ---------------------------------------------------------------- Test 1
def test1_reference_parity():
    """同一 prediction + 同一 GT：与参考实现 official_map（官方行）逐位一致。"""
    print("\n[Test 1] 与参考实现一致性")
    import official_map as OM
    IMG = PROJECT_ROOT / "data/processed/rgbid_split/images/val/visible"
    LBL = PROJECT_ROOT / "data/processed/rgbid_split/labels/val/visible"
    if not IMG.is_dir():
        ck("Test1 数据可用", False, "val split 缺失"); return
    per, _ = OE.load_split(IMG, LBL)
    for tag, rel in [("baseline", "diagnostic/rgbid_inference_gain/RGBID_baseline/results"),
                     ("Dprime", "diagnostic/sepstem_clahe/best_full/results")]:
        p = PROJECT_ROOT / rel
        if not p.is_dir():
            ck(f"Test1 {tag} 产物存在", False, rel); continue
        # 两者都走冻结协议（cap=100 + metric A）后应逐位一致
        g, pr, _ = OM.collect(IMG, LBL, p, 0.0, True, max_boxes=OE.MAX_BOXES_PER_IMAGE)
        ref = OM.evaluate(g, pr, avg="mean", tail="zero", match="conf", metric="A")["mAP50-95"]
        got = OE.evaluate(per, p, metric="A")["mAP50-95"]
        ck(f"Test1 {tag} 与参考一致", abs(ref - got) < 1e-12, f"ref={ref:.10f} got={got:.10f}")


# ---------------------------------------------------------------- Test 2
def test2_max_boxes():
    """>100 框：必须被按 conf 降序截断到恰好 100。"""
    print("\n[Test 2] 每图 100 框硬上限")
    rows = [[0, 0.5, 0.5, 0.1, 0.1, i / 1000.0] for i in range(150)]
    got = OE.apply_max_boxes(_mk(rows), OE.MAX_BOXES_PER_IMAGE)
    ck("Test2 截断到 100", len(got) == 100, f"len={len(got)}")
    ck("Test2 保留 conf 最高者", abs(got[0, 5] - 0.149) < 1e-6 and abs(got[-1, 5] - 0.050) < 1e-6,
       f"first={got[0,5]} last={got[-1,5]}")
    ck("Test2 conf 单调不增", bool(np.all(np.diff(got[:, 5]) <= 1e-12)))
    ck("Test2 常数 MAX_BOXES_PER_IMAGE == 100", OE.MAX_BOXES_PER_IMAGE == 100)


# ---------------------------------------------------------------- Test 3
def test3_invalid_rows():
    """NaN / 非有限 / 字段数错 / 类别越界：必须被拒绝且不计入。"""
    print("\n[Test 3] 非法行过滤")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.txt"
        f.write_text(
            "0 0.5 0.5 0.1 0.1 0.9\n"          # 合法
            "0 0.5 0.5 0.1 0.1 nan\n"          # NaN
            "0 0.5 0.5 0.1 0.1 inf\n"          # Inf
            "0 0.5 0.5 0.1\n"                  # 字段数错
            "99 0.5 0.5 0.1 0.1 0.5\n"         # 类别越界
            "0 0.5 0.5 0.1 0.1 0.8\n",         # 合法
            encoding="utf-8")
        n_total, valid, n_bad = OE.read_pred_txt(f)
        ck("Test3 总行数 6", n_total == 6, f"{n_total}")
        ck("Test3 合法行 2", len(valid) == 2, f"{len(valid)}")
        ck("Test3 非法行 4", n_bad == 4, f"{n_bad}")
        ck("Test3 无 NaN/Inf 残留", bool(np.isfinite(valid).all()))


# ---------------------------------------------------------------- Test 4
def test4_matching_order():
    """confidence 顺序：官方要求按 conf 降序贪心；高 conf 框必须优先拿到 GT。"""
    print("\n[Test 4] 匹配顺序 = confidence 降序")
    gt = {0: np.array([[0., 0., 10., 10.]])}
    # 高 conf 框 IoU=0.56（够 0.5 但差），低 conf 框与 GT 完全重合(IoU=1.0)。
    # 官方按 conf 序 → 高 conf 先拿走 GT，低 conf 变 FP。若按 IoU 序则结果相反。
    preds = [(0, 0.9, np.array([0., 0., 7., 8.])), (0, 0.1, np.array([0., 0., 10., 10.]))]
    rec, prec = OE.pr_curve_for_class(preds, gt, 0.5)
    ck("Test4 高 conf 先匹配(GT 被高 conf 拿走, 后者成 FP)",
       len(rec) == 2 and abs(rec[0] - 1.0) < 1e-9 and abs(prec[1] - 0.5) < 1e-6,
       f"rec={rec} prec={prec}")
    # 顺序无关性：同 conf 下重排不应改变结果
    p2 = list(reversed(preds))
    r2, _ = OE.pr_curve_for_class(p2, gt, 0.5)
    ck("Test4 conf 决定顺序而非输入顺序", abs(r2[-1] - rec[-1]) < 1e-12)


# ---------------------------------------------------------------- Test 5
def test5_denominator():
    """无 GT class：Metric-A 计入(AP=0)，Metric-B 排除 → A <= B。"""
    print("\n[Test 5] 无 GT 类的 denominator")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img = td / "im"; img.mkdir(); lab = td / "lb"; lab.mkdir()
        res = td / "rs"; res.mkdir()
        from PIL import Image
        Image.new("RGB", (100, 100)).save(img / "a.png")
        # 只有 class 0 有 GT；class 1 有预测无 GT
        (lab / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (res / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n1 0.3 0.3 0.1 0.1 0.8\n", encoding="utf-8")
        per, _ = OE.load_split(img, lab)
        A = OE.evaluate(per, res, metric="A")["mAP50-95"]
        B = OE.evaluate(per, res, metric="B")["mAP50-95"]
        ck("Test5 A < B（class1 无 GT）", A < B, f"A={A} B={B}")
        ck("Test5 B == 1.0（仅 class0，完全命中）", abs(B - 1.0) < 1e-9, f"B={B}")
        ck("Test5 A == 1/12（class0 满分, 其余 11 类记 0）", abs(A - 1.0 / 12) < 1e-9, f"A={A}")


# ---------------------------------------------------------------- Test 6
def test6_extreme_small():
    """极端小目标 / IoU=0：不得出现 NaN、负值或 >1 的 AP。"""
    print("\n[Test 6] 极端小目标与 IoU=0")
    gt = {0: np.array([[10., 10., 10.5, 10.5]])}
    # 完全不重叠
    rec, prec = OE.pr_curve_for_class([(0, 0.9, np.array([50., 50., 51., 51.]))], gt, 0.5)
    ap = OE.interpolate_ap(rec, prec)
    ck("Test6 IoU=0 → 有限且为 0", np.isfinite(ap) and abs(ap) < 1e-12, f"ap={ap}")
    # 退化框 w=h=0
    rec2, prec2 = OE.pr_curve_for_class([(0, 0.9, np.array([10., 10., 10., 10.]))], gt, 0.5)
    ap2 = OE.interpolate_ap(rec2, prec2)
    ck("Test6 退化框 → 有限且为 0", np.isfinite(ap2) and abs(ap2) < 1e-12, f"ap2={ap2}")
    # 亚像素目标，IoU=1
    gt2 = {0: np.array([[10., 10., 10.2, 10.2]])}
    rec3, prec3 = OE.pr_curve_for_class([(0, 0.9, np.array([10., 10., 10.2, 10.2]))], gt2, 0.95)
    ap3 = OE.interpolate_ap(rec3, prec3)
    ck("Test6 亚像素目标 IoU=1 → AP=1", abs(ap3 - 1.0) < 1e-9, f"ap3={ap3}")
    # 空预测
    ck("Test6 空预测 AP=0", OE.interpolate_ap(*OE.pr_curve_for_class([], gt, 0.5)) == 0.0)


if __name__ == "__main__":
    print("=" * 70)
    print("official_eval regression tests")
    print("=" * 70)
    test1_reference_parity()
    test2_max_boxes()
    test3_invalid_rows()
    test4_matching_order()
    test5_denominator()
    test6_extreme_small()
    print("\n" + "=" * 70)
    print(f"RESULT: {'ALL PASS' if not FAILS else 'FAIL -> ' + ', '.join(FAILS)}")
    print("=" * 70)
    sys.exit(1 if FAILS else 0)
