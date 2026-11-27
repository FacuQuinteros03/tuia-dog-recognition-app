"""Construye el conjunto independiente de evaluacion (ENUNCIADO 3.4) desde Wikimedia Commons.

Se eligio Commons porque cada archivo trae autor y licencia (CC0, CC BY, CC BY-SA o dominio
publico), lo que permite documentar la fuente y la licencia de cada imagen (ENUNCIADO 3.8).

Modos:
    candidates  Busca fotos con licencia libre por raza / escena, sortea candidatas con semilla
                fija, las descarga en data/dataset/independent_candidates/ y escribe
                data/dataset/independent_candidates.csv (una fila por imagen, columna keep=1).
                Despues de revisar las imagenes a mano se marca keep=0 (con el motivo en
                `reason`) en las que no sirven.
    finalize    Toma las candidatas con keep=1 (hasta --per-breed por raza), las copia a
                data/dataset/independent/<raza>/ y data/dataset/independent_scenes/<escena>/
                y escribe data/independent_sources.csv, que es la lista definitiva.
    download    Reconstruye el conjunto desde data/independent_sources.csv (para un clon limpio).
    package     Arma output/conjunto_independiente.zip para compartir con los docentes: las
                imagenes, el CSV de fuentes y ATRIBUCIONES.md (autor y licencia de cada imagen,
                que CC BY y CC BY-SA exigen tambien al redistribuir).

Las imagenes quedan fuera del control de versiones (data/dataset/* esta en el .gitignore);
solo se versiona el CSV de fuentes y licencias.

Uso:
    python scripts/build_independent_set.py candidates
    python scripts/build_independent_set.py finalize --per-breed 5
    python scripts/build_independent_set.py download
    python scripts/build_independent_set.py package
"""
from __future__ import annotations

import argparse
import csv
import html
import random
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CANDIDATES_DIR = DATA / "dataset" / "independent_candidates"
BREEDS_DIR = DATA / "dataset" / "independent"
SCENES_DIR = DATA / "dataset" / "independent_scenes"
CANDIDATES_CSV = DATA / "dataset" / "independent_candidates.csv"  # archivo de trabajo, no versionado
SOURCES_CSV = DATA / "independent_sources.csv"
PACKAGE_ZIP = ROOT / "output" / "conjunto_independiente.zip"

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "tuia-cv-tp2-student-project/1.0 (educational, low volume)"
THUMB_WIDTH = 800
SEED = 42

# Raza del dataset (nombre de carpeta) -> categoria de Commons (se busca en todo su arbol).
# Se eligieron 24 razas, incluyendo pares faciles de confundir dentro del dataset:
# Labrador/Golden, Collie/Border Collie, Malinois/German Sheperd, Bulldog/French Bulldog,
# Maltese/Bichon Frise, Shih-Tzu/Lhasa.
BREEDS: dict[str, str] = {
    "Beagle": "Beagle (dog)",
    "Boxer": "Boxer (dog)",
    "Border Collie": "Border Collie",
    "Collie": "Rough Collie",
    "Chihuahua": "Chihuahua (dog)",
    "Golden Retriever": "Golden Retriever",
    "Labrador": "Labrador Retriever",
    "Siberian Husky": "Siberian Husky",
    "Shiba Inu": "Shiba Inu",
    "Pug": "Pug",
    "Dalmation": "Dalmatian spotted dogs",
    "Corgi": "Pembroke Welsh Corgi",
    "German Sheperd": "German Shepherd Dog",
    "Malinois": "Belgian Shepherd Malinois",
    "Poodle": "Poodles",
    "Rottweiler": "Rottweiler",
    "Doberman": "Dobermann",
    "Bulldog": "Bulldog",
    "French Bulldog": "French Bulldog",
    "Maltese": "Maltese dog",
    "Bichon Frise": "Bichon Frisé",
    "Shih-Tzu": "Shih Tzu",
    "Lhasa": "Lhasa Apso",
    "Dingo": "Dingoes",
}

# Escenas para la Etapa 3: varios perros y ningun perro (gatos y zorros, para falsos positivos).
SCENES: dict[str, list[str]] = {
    "multi_dog": ["Groups of dogs"],
    "no_dog": ["Sleeping cats", "Vulpes vulpes"],
}

N_CANDIDATES = {"breed": 9, "multi_dog": 30, "no_dog": 25}

ALLOWED_LICENSE = re.compile(r"^(cc0|cc[- ]by(-sa)?[- ]\d|public domain|pd)", re.IGNORECASE)
# Palabras en el titulo que suelen indicar que no es una foto de un solo perro real.
EXCLUDE_WORDS = re.compile(
    r"\b(drawing|illustration|painting|paintings|art|stamp|statue|sculpture|skull|skeleton|logo|"
    r"map|postcard|book|engraving|coin|toy|plush|cartoon|poster|figurine|emoji|icon|"
    r"man|woman|girl|boy|child|children|kid|people|owner|family|crowd)\b",
    re.IGNORECASE,
)
EXCLUDE_WORDS_BREED = re.compile(r"\b(puppies|dogs|and|with|pack|litter)\b", re.IGNORECASE)

FIELDS = [
    "group", "label", "file", "commons_title", "page_url", "image_url", "author",
    "license", "license_url", "category", "keep", "reason",
]

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def api(**params: str | int) -> dict:
    """Llamada a la API de Commons con pausa entre requests y reintentos ante rate limit."""
    params.update(action="query", format="json", maxlag=5)
    last = None
    for attempt in range(6):
        r = session.get(API, params=params, timeout=30)
        last = r
        if r.status_code == 200 and r.text.startswith("{"):
            data = r.json()
            if "error" not in data:
                time.sleep(1.0)
                return data
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"La API de Commons fallo: {last.status_code if last else '?'}")


def search_files(category: str, limit: int = 300) -> list[str]:
    """Titulos de archivos bitmap en todo el arbol de la categoria (CirrusSearch deepcat)."""
    data = api(
        list="search", srnamespace=6, srlimit=limit,
        srsearch=f'filetype:bitmap deepcat:"{category}"',
    )
    return [hit["title"] for hit in data["query"]["search"]]


def image_infos(titles: list[str]) -> list[dict]:
    """Metadata (url de la miniatura, autor, licencia) de hasta 50 archivos por request."""
    out = []
    for i in range(0, len(titles), 50):
        data = api(
            titles="|".join(titles[i : i + 50]), prop="imageinfo",
            iiprop="url|size|mime|user|extmetadata", iiurlwidth=THUMB_WIDTH,
            iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist",
        )
        for page in data["query"].get("pages", {}).values():
            if "imageinfo" not in page:
                continue
            info = page["imageinfo"][0]
            meta = info.get("extmetadata", {})
            author = re.sub(r"<[^>]+>", "", html.unescape(meta.get("Artist", {}).get("value", "")))
            if not author.strip() and info.get("user"):
                # Sin campo Artist: se atribuye al usuario que subio el archivo, como hace Commons.
                author = f"{info['user']} (usuario de Wikimedia Commons)"
            out.append({
                "commons_title": page["title"],
                "page_url": info.get("descriptionurl", ""),
                "image_url": info.get("thumburl") or info.get("url", ""),
                "mime": info.get("mime", ""),
                "width": info.get("width", 0),
                "author": " ".join(author.split()),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "license_url": meta.get("LicenseUrl", {}).get("value", ""),
            })
    return out


def is_usable(info: dict, group: str) -> bool:
    title = info["commons_title"].lower()
    if info["mime"] != "image/jpeg" or info["width"] < 400:
        return False
    if not ALLOWED_LICENSE.match(info["license"].strip()):
        return False
    if EXCLUDE_WORDS.search(title):
        return False
    if group == "breed" and EXCLUDE_WORDS_BREED.search(title):
        return False
    return True


def slug(text: str) -> str:
    stem = Path(text.removeprefix("File:")).stem
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:60]


def download(url: str, dest: Path) -> bool:
    """Descarga respetando el rate limit de Wikimedia (429 + Retry-After). Devuelve si lo logro."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    wait = 10.0
    for _ in range(8):
        r = session.get(url, timeout=60)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            time.sleep(1.5)
            return True
        retry_after = r.headers.get("Retry-After", "")
        time.sleep(float(retry_after) if retry_after.isdigit() else wait)
        wait = min(wait * 2, 120.0)
    print(f"  No se pudo descargar {url}: {r.status_code}", flush=True)
    return False


def cmd_candidates() -> None:
    """Fase 1: elige candidatas y escribe el CSV. Fase 2: descarga las que faltan (reanudable)."""
    if not CANDIDATES_CSV.exists():
        rng = random.Random(SEED)
        jobs = [("breed", breed, [cat]) for breed, cat in BREEDS.items()]
        jobs += [(scene, scene, cats) for scene, cats in SCENES.items()]
        rows = []
        for group, label, categories in jobs:
            titles: list[str] = []
            for cat in categories:
                titles += search_files(cat)
            titles = sorted(set(titles))
            usable = [i for i in image_infos(titles) if is_usable(i, group)]
            usable.sort(key=lambda i: i["commons_title"])
            picked = rng.sample(usable, min(N_CANDIDATES[group], len(usable)))
            print(f"{label}: {len(titles)} archivos, {len(usable)} usables, {len(picked)} candidatas", flush=True)
            for info in picked:
                file = Path(group if group != "breed" else "breeds") / label / f"{slug(info['commons_title'])}.jpg"
                rows.append({
                    "group": group, "label": label, "file": file.as_posix(),
                    "commons_title": info["commons_title"], "page_url": info["page_url"],
                    "image_url": info["image_url"], "author": info["author"],
                    "license": info["license"], "license_url": info["license_url"],
                    "category": "; ".join(categories), "keep": 1, "reason": "",
                })
        write_csv(CANDIDATES_CSV, rows)
    rows = read_csv(CANDIDATES_CSV)
    for row in rows:
        dest = CANDIDATES_DIR / row["file"]
        if not dest.exists() and str(row["keep"]) == "1" and not download(row["image_url"], dest):
            row["keep"], row["reason"] = 0, "descarga fallida"
    write_csv(CANDIDATES_CSV, rows)
    print(f"{len(rows)} candidatas en {CANDIDATES_DIR} y {CANDIDATES_CSV}")


def final_path(row: dict) -> Path:
    name = Path(row["file"]).name
    if row["group"] == "breed":
        return BREEDS_DIR / row["label"] / name
    return SCENES_DIR / row["label"] / name


def cmd_finalize(per_breed: int) -> None:
    rows = read_csv(CANDIDATES_CSV)
    kept, count = [], {}
    for row in rows:
        if str(row["keep"]) != "1":
            continue
        key = row["label"]
        if row["group"] == "breed" and count.get(key, 0) >= per_breed:
            continue
        count[key] = count.get(key, 0) + 1
        dest = final_path(row)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CANDIDATES_DIR / row["file"], dest)
        kept.append({k: v for k, v in row.items() if k not in ("keep", "reason")})
    for d in (BREEDS_DIR, SCENES_DIR):
        for f in d.rglob("*.jpg"):
            if f not in {final_path(r) for r in kept}:
                f.unlink()
    write_csv(SOURCES_CSV, kept, [f for f in FIELDS if f not in ("keep", "reason")])
    short = [k for k, v in count.items() if k in BREEDS and v < per_breed]
    print(f"{len(kept)} imagenes finales -> {SOURCES_CSV}")
    if short:
        print(f"Razas con menos de {per_breed}: {short}")


def cmd_download() -> None:
    rows = read_csv(SOURCES_CSV)
    for row in rows:
        dest = final_path(row)
        if not dest.exists():
            download(row["image_url"], dest)
    print(f"{len(rows)} imagenes en {BREEDS_DIR} y {SCENES_DIR}")


def cmd_package() -> None:
    rows = read_csv(SOURCES_CSV)
    missing = [str(final_path(r)) for r in rows if not final_path(r).exists()]
    if missing:
        sys.exit(f"Faltan {len(missing)} imagenes (correr el modo download): {missing[:3]}")
    lines = [
        "# Conjunto independiente de evaluacion - atribuciones",
        "",
        "Imagenes de Wikimedia Commons, redistribuidas sin modificar (miniatura de 800 px de ancho)",
        "solo para la evaluacion del TP2 de Computer Vision (TUIA, FCEIA-UNR).",
        "La pagina de cada archivo en Commons tiene el detalle completo de autor y licencia.",
        "",
        "| Archivo | Autor | Licencia | Fuente |",
        "|---|---|---|---|",
    ]
    for r in rows:
        rel = final_path(r).relative_to(DATA / "dataset").as_posix()
        author = r["author"].replace("|", "/") or "desconocido"
        lic = f"[{r['license']}]({r['license_url']})" if r["license_url"] else r["license"]
        lines.append(f"| `{rel}` | {author} | {lic} | [{r['commons_title']}]({r['page_url']}) |")
    PACKAGE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PACKAGE_ZIP, "w", zipfile.ZIP_STORED) as zf:  # JPEG ya esta comprimido
        for r in rows:
            path = final_path(r)
            zf.write(path, path.relative_to(DATA / "dataset").as_posix())
        zf.write(SOURCES_CSV, SOURCES_CSV.name)
        zf.writestr("ATRIBUCIONES.md", "\n".join(lines) + "\n")
    size = PACKAGE_ZIP.stat().st_size / 1e6
    print(f"{len(rows)} imagenes -> {PACKAGE_ZIP} ({size:.1f} MB)")
    print("Descomprimir en data/dataset/ (crea independent/ e independent_scenes/).")


def write_csv(path: Path, rows: list[dict], fields: list[str] = FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"No existe {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("candidates", "finalize", "download", "package"))
    parser.add_argument("--per-breed", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "candidates":
        cmd_candidates()
    elif args.mode == "finalize":
        cmd_finalize(args.per_breed)
    elif args.mode == "package":
        cmd_package()
    else:
        cmd_download()


if __name__ == "__main__":
    main()
