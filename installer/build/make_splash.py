"""Draws build/splash.bmp, the picture Windows shows while the portable installer unpacks itself.

    python build/make_splash.py            (reads the versions from package.json)

The look matches electron/splash.html, so the hand-over to the installer window is seamless."""
import json
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
W, H = 520, 300
BG = (11, 18, 32)
TEAL = (20, 184, 166)


def font(names, size):
    dirs = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu", "C:\\Windows\\Fonts"]
    for n in names:
        for d in dirs:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    pkg = json.load(open(os.path.join(ROOT, "package.json"), encoding="utf-8"))
    apsi_v, setup_v = pkg["version"], pkg.get("setupVersion", pkg["version"])
    im = Image.new("RGB", (W, H), BG)
    # a soft teal glow behind the wordmark
    glow = Image.new("L", (W, H), 0)
    ImageDraw.Draw(glow).ellipse((W // 2 - 150, 40, W // 2 + 150, 190), fill=60)
    glow = glow.filter(ImageFilter.GaussianBlur(40))
    im = Image.composite(Image.new("RGB", (W, H), (16, 52, 60)), im, glow)
    mark = Image.open(os.path.join(ROOT, "public", "wordmark.png")).convert("RGBA")
    mw = 200
    mark = mark.resize((mw, round(mark.height * mw / mark.width)), Image.LANCZOS)
    im.paste(mark, ((W - mw) // 2, 46), mark)
    d = ImageDraw.Draw(im)
    f1 = font(["Inter-Medium.otf", "segoeui.ttf", "DejaVuSans.ttf"], 16)
    f2 = font(["Inter-Regular.otf", "segoeui.ttf", "DejaVuSans.ttf"], 12)
    y = 46 + mark.height + 18
    for text, f, col, dy in ((f"Preparing Apsi {apsi_v} Setup…", f1, (203, 213, 225), 0),
                             (f"Setup {setup_v}  ·  unpacking, this takes a few seconds", f2, (100, 116, 139), 28)):
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y + dy), text, font=f, fill=col)
    d.rectangle((0, H - 4, W, H), fill=(15, 52, 56))
    d.rectangle((0, H - 4, int(W * 0.62), H), fill=TEAL)
    im.save(os.path.join(HERE, "splash.bmp"))
    print(f"build/splash.bmp: Apsi {apsi_v}, Setup {setup_v}")


if __name__ == "__main__":
    main()
