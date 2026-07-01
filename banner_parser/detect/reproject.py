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
