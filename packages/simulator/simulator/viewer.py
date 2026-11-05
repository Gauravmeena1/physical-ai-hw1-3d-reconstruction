"""pygame preview panels. The ONLY module in the package that imports pygame.

On Linux, SDL must be forced to a pure-software X11 window BEFORE
`import pygame`, or pygame's GLX context collides with habitat's GL and
crashes with `X Error ... X_GLXMakeCurrent BadAccess` (see hw1
root_cause_analysis). On macOS there is no X11, so the SDL overrides are
skipped and pygame uses its default Cocoa driver. Callers must also
construct Engine BEFORE initializing the viewer window, and must never
import pygame directly — always through this module.

PERFORMANCE: `Preview` is the incremental painter load.py drives. It keeps a
copy of every panel as last painted, re-blits only the panels whose pixels
changed, and pushes only those rects to the window (display.update(rects)
instead of a full flip). Standing still outside a flicker zone only the depth
panel (per-frame depth noise) is touched; the RGB and bird's-eye panels cost a
memcmp each. Besides the CPU saved, this shrinks the damaged window area that
remote desktops (NX / VNC / RDP) must re-encode every frame. `draw()` is the
stateless full-repaint variant with the original signature.
"""

import os
import sys

# Linux/X11 only: force a pure-software X11 window BEFORE `import pygame`, or
# pygame's GLX context collides with habitat's GL and crashes with
# `X Error ... X_GLXMakeCurrent BadAccess` (see hw1 root_cause_analysis).
# On macOS there is no X11 — SDL must use its default Cocoa video driver, so
# SDL_VIDEODRIVER must NOT be set (setting it to "x11" makes pygame.init()
# fail with "x11 not available"). SDL_RENDER_DRIVER=software is harmless on
# both, but only force it on Linux to leave macOS defaults alone.
if sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")

import pygame  # noqa: E402  (env vars above must precede this import)

import numpy as np  # noqa: E402
import cv2  # noqa: E402

# Window-system events after which the whole window must be repainted: the
# incremental Preview pushes only changed panels, so an expose / restore would
# otherwise leave stale or blank regions until those panels next change.
EXPOSE_EVENTS = tuple(
    getattr(pygame, name) for name in
    ("VIDEOEXPOSE", "WINDOWEXPOSED", "WINDOWSHOWN", "WINDOWRESTORED",
     "WINDOWSIZECHANGED", "WINDOWMAXIMIZED")
    if hasattr(pygame, name))


def panel_keys(display_cfg):
    """Frame keys of the enabled panels, left to right."""
    keys = []
    if display_cfg.get("show_rgb", True):
        keys.append("rgb")
    if display_cfg.get("show_depth", True):
        keys.append("depth_vis")
    if display_cfg.get("show_birdseye", True):
        keys.append("birdseye")
    return keys or ["rgb"]


def panels_of(frame, display_cfg):
    """Enabled panels as (H, W, 3) RGB uint8 arrays, heights matched to the first."""
    panels = [frame[k] for k in panel_keys(display_cfg)]
    # Panels may have different resolutions; match heights before placing.
    h = panels[0].shape[0]
    return [
        p if p.shape[0] == h
        else cv2.resize(p, (int(round(p.shape[1] * h / p.shape[0])), h))
        for p in panels
    ]


def build_canvas(frame, display_cfg):
    """Stack the enabled panels (first-person RGB + depth + bird's-eye)
    horizontally into one (H, W, 3) RGB image."""
    return np.concatenate(panels_of(frame, display_cfg), axis=1)


def draw_counter(screen, count, font, fps=None):
    """Hover text overlay (top-left): captured-frame count, plus the preview
    frame rate when `fps` is given."""
    text = f"Captured frames: {count}"
    if fps is not None:
        text += f"   {fps:.0f} fps"
    label = font.render(text, True, (255, 255, 0))
    pad = 6
    bg = pygame.Surface((label.get_width() + 2 * pad, label.get_height() + 2 * pad))
    bg.set_alpha(140)
    bg.fill((0, 0, 0))
    screen.blit(bg, (8, 8))
    screen.blit(label, (8 + pad, 8 + pad))


def blit_panel(screen, panel, rect):
    """Paint one (H, W, 3) RGB array into `rect` of `screen`, scaling if needed."""
    # pygame surfaces are (W, H, 3); our arrays are (H, W, 3) -> swap axes 0/1.
    arr = np.transpose(panel, (1, 0, 2))
    if rect.size == (panel.shape[1], panel.shape[0]) and screen.get_rect().contains(rect):
        pygame.surfarray.blit_array(screen.subsurface(rect), arr)   # no temp Surface
    else:
        surf = pygame.transform.scale(pygame.surfarray.make_surface(arr), rect.size)
        screen.blit(surf, rect.topleft)


class Preview:
    """Incremental panel painter (see module docstring).

    draw(frame, count, fps=None) paints the enabled panels of `frame` left to
    right, redrawing only those whose pixels differ from what is on screen, and
    updates only those window rects. The overlay sits on the first panel, so a
    count change (or an fps refresh, throttled to once a second) repaints that
    panel. invalidate() forces a full repaint on the next draw — call it on
    window expose / restore events (EXPOSE_EVENTS)."""

    FPS_REFRESH_MS = 1000

    def __init__(self, screen, display_cfg, font):
        self.screen = screen
        self.display_cfg = display_cfg
        self.font = font
        self.scale = float(display_cfg["scale"])
        self._last = []           # per panel: private copy of what is on screen
        self._count = None
        self._fps_shown = None    # fps value currently painted (rounded)
        self._fps_at = None       # pygame ticks of the last fps refresh

    def invalidate(self):
        """Repaint every panel on the next draw."""
        self._last = []

    def _rects(self, panels):
        """Window rect per panel — cumulative source width scaled, so the layout
        matches build_canvas scaled as a whole (what sized the window)."""
        rects, x_src = [], 0
        for p in panels:
            x0 = int(x_src * self.scale)
            x_src += p.shape[1]
            x1 = int(x_src * self.scale)
            rects.append(pygame.Rect(x0, 0, x1 - x0, int(p.shape[0] * self.scale)))
        return rects

    def draw(self, frame, count, fps=None):
        panels = panels_of(frame, self.display_cfg)
        if len(self._last) != len(panels):
            self._last = [None] * len(panels)

        fps_shown = self._fps_shown
        if fps is not None:
            now = pygame.time.get_ticks()
            if self._fps_at is None or now - self._fps_at >= self.FPS_REFRESH_MS:
                fps_shown, self._fps_at = int(round(float(fps))), now
        overlay_dirty = count != self._count or fps_shown != self._fps_shown

        dirty = []
        for i, (p, rect) in enumerate(zip(panels, self._rects(panels))):
            prev = self._last[i]
            changed = (prev is None or prev.shape != p.shape
                       or not np.array_equal(prev, p))
            if i == 0 and overlay_dirty:
                changed = True      # overlay is composited onto this panel
            if not changed:
                continue
            blit_panel(self.screen, p, rect)
            # Private copy (~0.1 ms, changed panels only): equality against a
            # buffer the producer might later overwrite in place would report
            # "unchanged" for a frame that did change. habitat allocates fresh
            # readouts today; the copy keeps that from ever mattering.
            self._last[i] = np.array(p, copy=True)
            dirty.append(rect)
        if dirty and dirty[0].x == 0:
            # First panel was repainted -> overlay must go back on top of it.
            draw_counter(self.screen, count, self.font, fps_shown)
        self._count, self._fps_shown = count, fps_shown
        if dirty:
            pygame.display.update(dirty)


def draw(screen, frame, display_cfg, count, font):
    """Stateless full repaint of every enabled panel + overlay (original API)."""
    Preview(screen, display_cfg, font).draw(frame, count)
