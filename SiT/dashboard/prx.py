"""Bind the dashboard to the user's installed PRX template without modifying its service."""
from hashlib import sha256
from html import escape
import re

EXPECTED = "898b6993ff8e40e7a6a9eb93c2344acabd37e90507d97d8b619f4cc57a871198"


def between(text, start, end):
    a = text.index(start)
    return text[a:text.index(end, a + len(start))]


def tag(text, begin, end):
    a = text.index(begin)
    return text[a:text.index(end, a) + len(end)]


def build(path, page):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("PRX template changed; compare it before adapting")
    src = raw.decode()
    header = tag(src, '<header class="top">', '</header>')
    for identifier, value in (("eyebrow", "SERVER 67 · IMAGENET 256"), ("pageTitle", "VA-Diffusion · SiT-S/2")):
        header = re.sub(r'(id="' + identifier + r'"[^>]*>)[^<]*', lambda m: m[1]+escape(value), header)
    fragments = {"STYLE": tag(src, "<style>", "</style>"), "HEADER": header,
                 "SAVES": tag(src, '<section class="checkpointStrip">', '</section>'),
                 "LIGHTBOX": between(src, '<div id="lightbox" class="lightbox"', '<div id="videoModal"').strip()}
    for key, value in fragments.items():
        page = page.replace("{{" + key + "}}", value)
    common = between(src, "const $ = selector", "const selectorVersion")
    charts = between(src, "function pills(items)", "function countForStep")
    steps = between(src, "function renderStepTools()", "  const hasCurrentResult") + "}\n"
    steps = steps.replace("STEP 0 회귀 기준선 결과 대기", "STEP 0 기준선 대기")
    lightbox = between(src, "function showLightbox(index)", "function openSample")
    saves = between(src, "function renderCheckpoint(", "function render(d)")
    saves = saves.replace("latest.pt 저장", "addon.safetensors").replace("1시간 주기 · latest + best · 테스트",
                                                                        "30분 주기 · addon 가중치 · 샘플 36장")
    return page.encode(), (common + charts + steps + lightbox + saves).encode()
