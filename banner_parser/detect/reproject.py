"""Репроекция equirectangular-панорамы в плоские перспективные виды.

Зачем: на equirect объекты по краям сильно искажены («загибаются»), и детектор
на них работает плохо. Мы нарезаем панораму на несколько неискажённых видов
по горизонту, детектим на них, а bbox возвращаем обратно в нормализованные
координаты панорамы через карту uv_map.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PerspectiveView:
    image: Image.Image
    uv_map: np.ndarray   # (out_h, out_w, 2) → (fx, fy) в equirect [0..1]


def equirect_to_perspective(equi: Image.Image, yaw_deg: float, pitch_deg: float,
                            fov_h_deg: float = 90.0, out_w: int = 1024,
                            out_h: int = 1024) -> PerspectiveView:
    E = np.asarray(equi)
    H, W = E.shape[:2]
    vfov_rad = np.radians(360.0 * H / W)   # вертикальный охват equirect

    f = (out_w / 2) / np.tan(np.radians(fov_h_deg) / 2)
    xs = np.arange(out_w) - out_w / 2 + 0.5
    ys = np.arange(out_h) - out_h / 2 + 0.5
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, f)
    vec = np.stack([xx, -yy, zz], axis=-1).astype(np.float64)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    # поворот вокруг X (pitch), затем вокруг Y (yaw)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    R = Ry @ Rx
    world = vec @ R.T

    A = np.arctan2(world[..., 0], world[..., 2])          # азимут в кадре панорамы
    Elev = np.arcsin(np.clip(world[..., 1], -1, 1))       # возвышение
    fx = (A / (2 * np.pi)) % 1.0
    fy = np.clip(0.5 - Elev / vfov_rad, 0.0, 1.0)

    map_x = np.clip((fx * W).astype(np.int64), 0, W - 1)
    map_y = np.clip((fy * H).astype(np.int64), 0, H - 1)
    persp = E[map_y, map_x]
    uv = np.stack([fx, fy], axis=-1)
    return PerspectiveView(Image.fromarray(persp), uv)


def horizon_views(equi: Image.Image, n: int = 6, fov_h_deg: float = 90.0,
                  pitch_deg: float = 0.0, out: int = 1024) -> list[tuple[float, PerspectiveView]]:
    """n перекрывающихся видов по кругу горизонта. Возвращает (yaw, view)."""
    step = 360.0 / n
    return [(i * step, equirect_to_perspective(equi, i * step, pitch_deg,
                                                fov_h_deg, out, out))
            for i in range(n)]


def rectify_region(crop: Image.Image, fx0: float, fy0: float, fx1: float, fy1: float,
                   pano_w: int, pano_h: int, margin: float = 0.06,
                   max_side: int = 1600) -> Image.Image:
    """Выпрямить equirect-вырезку в перспективную (неискажённую) картинку.

    Зачем: бокс детектора приходит из перспективного вида, а кроп режется по
    осепараллельному прямоугольнику в equirect. Обратное отображение кривой
    области в прямоугольник раздувает бокс и заваливает текст трапецией —
    именно на таком кропе OCR и ломается. Здесь мы отображаем вырезку обратно
    в перспективу, центрируя камеру на середине бокса.

    crop покрывает панораму в долях [fx0..fx1] x [fy0..fy1]; pano_w/pano_h —
    размеры ПОЛНОЙ панорамы того же зума (нужны для вертикального охвата).
    """
    cw, ch = crop.size
    if cw < 8 or ch < 8:
        return crop
    vfov = 2.0 * np.pi * pano_h / pano_w        # вертикальный охват панорамы, рад

    # Угловые границы бокса и центр камеры.
    a0, a1 = fx0 * 2.0 * np.pi, fx1 * 2.0 * np.pi
    e0, e1 = (0.5 - fy1) * vfov, (0.5 - fy0) * vfov
    da, de = (a1 - a0) * (1.0 + 2 * margin), (e1 - e0) * (1.0 + 2 * margin)
    if da <= 0 or de <= 0:
        return crop
    da = min(da, np.radians(120.0))             # шире 120° перспектива вырождается
    ac, ec = (a0 + a1) / 2.0, (e0 + e1) / 2.0

    # Размер выхода: сохраняем разрешение исходной вырезки, но не раздуваем.
    ow = int(min(max_side, max(32, cw)))
    oh = int(min(max_side, max(32, round(ow * (de / da)))))

    f = (ow / 2.0) / np.tan(da / 2.0)
    xs = np.arange(ow) - ow / 2.0 + 0.5
    ys = np.arange(oh) - oh / 2.0 + 0.5
    xx, yy = np.meshgrid(xs, ys)
    vec = np.stack([xx, -yy, np.full_like(xx, f)], axis=-1)
    vec /= np.linalg.norm(vec, axis=-1, keepdims=True)

    # Поворот на центр бокса: сначала подъём (pitch), затем азимут (yaw).
    cp, sp = np.cos(ec), np.sin(ec)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cy, sy = np.cos(ac), np.sin(ac)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    world = vec @ (Ry @ Rx).T

    A = np.arctan2(world[..., 0], world[..., 2])
    E = np.arcsin(np.clip(world[..., 1], -1, 1))
    fx = (A / (2 * np.pi)) % 1.0
    fy = 0.5 - E / vfov

    # Панорама замкнута по кругу: разворачиваем fx в окрестность бокса.
    fx = np.where(fx - fx0 > 0.5, fx - 1.0, fx)
    fx = np.where(fx0 - fx > 0.5, fx + 1.0, fx)

    # Доли панорамы -> пиксели вырезки.
    sx = (fx - fx0) / max(1e-9, fx1 - fx0) * (cw - 1)
    sy = (fy - fy0) / max(1e-9, fy1 - fy0) * (ch - 1)
    valid = (sx >= 0) & (sx <= cw - 1) & (sy >= 0) & (sy <= ch - 1)
    sx = np.clip(sx, 0, cw - 1).astype(np.int64)
    sy = np.clip(sy, 0, ch - 1).astype(np.int64)

    src = np.asarray(crop.convert("RGB"))
    out = src[sy, sx]
    out[~valid] = 0
    return Image.fromarray(out)
