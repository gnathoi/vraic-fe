"""Build the versioned data pack from the official raw files.

python -m jfe.ingest --raw data/raw --out data/processed [--pin]

Every derived parameter is computed here, once, from the checksummed sources listed in
data/manifests/sources.json. Nothing at runtime touches the network.
"""
import argparse
import csv
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

CODES = ["L2", "L1", "MD", "H1", "H2"]  # official fertility / life-expectancy assumption codes
MIG = [0, 200, 400, 600, 800]  # official net-migration scenarios
MIG_FILES = {0: "S07-nil", 200: "S07-200", 400: "S07", 600: "S07-600", 800: "S07-800"}
A = 101  # single years 0..99 plus open-ended 100+
SEX = ["F", "M"]
LTC_BANDS = [("65-74", 65, 74), ("75-84", 75, 84), ("85-94", 85, 94), ("95+", 95, 100)]
FERT_TRAIN = [0, 400, 800]  # fertility shape is fitted on these; +200/+600 are held out
PACK_VERSION = "jfe-pack-1"


def school_weights(first_group_age, groups):
    """Weights over single ages at 31 December for a run of school year groups.
    A year group starting school at age `a` in September contains children aged a (born Jan-Aug, 8/12)
    and a+1 (born Sep-Dec, 4/12) at the year end, assuming births spread evenly across months."""
    w = np.zeros(A)
    for g in range(groups):
        a = first_group_age + g
        w[a] += 8 / 12
        w[a + 1] += 4 / 12
    return w


W_PRIMARY = school_weights(4, 7)  # Reception (age 4 in September) to Year 6
W_SECONDARY = school_weights(11, 7)  # Year 7 to Year 13 (secondary schools include sixth forms)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(sources, raw, pin, manifest_path, manifest):
    raw.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in sources:
        p = raw / s["file"]
        if not p.exists():
            print(f"downloading {s['id']} {s['url']}")
            with urllib.request.urlopen(s["url"], timeout=120) as r:
                p.write_bytes(r.read())
        head = p.read_bytes()[:200].lower()
        if b"<html" in head or b"<!doctype" in head:
            raise SystemExit(f"{p} is an HTML page, not CSV")
        digest, size = sha256(p), p.stat().st_size
        if s.get("sha256") and s["sha256"] != digest:
            raise SystemExit(f"checksum mismatch for {s['id']}: {digest} != pinned {s['sha256']}")
        if pin:
            s["sha256"], s["bytes"] = digest, size
            s["retrieved"] = s.get("retrieved") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))
        out[s["id"]] = p
    if pin:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def age_forward(n):
    """Age everyone one year; the open-ended 100+ group keeps its survivors."""
    o = np.zeros_like(n)
    o[..., 1:] = n[..., :-1]
    o[..., -1] += n[..., -1]
    return o


def read_projection(path):
    """-> years, N[f,m,Y,2,A], D[f,m,Y,2,A], B[f,m,Y,2] (births by child's sex, on the age-0 row)."""
    rows = list(csv.DictReader(open(path, newline="")))
    years = sorted({int(r["Year"]) for r in rows})
    Y = len(years)
    N = np.full((5, 5, Y, 2, A), np.nan)
    D = np.full_like(N, np.nan)
    B = np.full((5, 5, Y, 2), np.nan)
    for r in rows:
        f, m = CODES.index(r["Fertility"]), CODES.index(r["Life expectancy"])
        t, a, s = years.index(int(r["Year"])), int(r["Age"]), SEX.index(r["Sex"])
        N[f, m, t, s, a] = float(r["Population"])
        D[f, m, t, s, a] = float(r["Deaths"] or 0)
        if a == 0:
            B[f, m, t, s] = float(r["Births"])
    assert not np.isnan(N).any() and not np.isnan(B).any(), f"{path}: incomplete cube"
    return years, N, D, B


def migration_residual(N, D, B):
    """Net migrants per cell implied by the official event order: ageing -> deaths -> births -> migration."""
    post = age_forward(N[..., :-1, :, :]) - D[..., 1:, :, :]
    post[..., 0] = B[..., 1:, :]
    return N[..., 1:, :, :] - post


def published_shape(table):
    """Published 2023-25 Jersey age-specific fertility rates, applied uniformly within each 5-year band.
    Scaled so that intensity 1.0 reproduces the published 2023-25 rates exactly."""
    v = np.zeros(A)
    for band, w, b in zip(table["bands"], table["women"], table["births"]):
        lo, hi = map(int, band.split("-"))
        v[lo:hi + 1] = b / w
    return v


def _intensity(sh, exposure, births, keys):
    num = sum(births[k] * (exposure[k] @ sh) for k in keys)
    den = sum((exposure[k] @ sh) ** 2 for k in keys)
    return num / den


def read_estimates(path):
    rows = list(csv.DictReader(open(path, newline="")))
    years = sorted({int(r["Year"]) for r in rows})
    obs = np.full((len(years), 2, A), np.nan)
    for r in rows:
        t, a = years.index(int(r["Year"])), int(r["Age"])
        for s, col in enumerate(["Female", "Male"]):
            if r[col].strip():
                obs[t, s, a] = float(r[col])
    return years, obs


def read_ltc(path, year=2024):
    claims = np.full((len(LTC_BANDS), 2), np.nan)
    status = {}
    for r in csv.DictReader(open(path, newline="")):
        if not r["Year_Ending"].endswith(str(year)) or r["Age_Band"] not in [b[0] for b in LTC_BANDS]:
            continue
        b, s = [x[0] for x in LTC_BANDS].index(r["Age_Band"]), SEX.index(r["Sex"])
        v = r["Total_Number_of_Claims_at_Year_End"].strip()
        status[f"{r['Age_Band']}/{r['Sex']}"] = "suppressed" if v.startswith("<") else "published"
        if not v.startswith("<"):
            claims[b, s] = float(v)
    return claims, status


def build(raw, out, pin=False, manifest_path=Path("data/manifests/sources.json")):
    manifest = json.loads(manifest_path.read_text())
    files = fetch(manifest["sources"], raw, pin, manifest_path, manifest)
    q = {"checks": []}

    def check(name, ok, **detail):
        q["checks"].append({"check": name, "pass": bool(ok), **detail})
        if not ok:
            raise SystemExit(f"data check failed: {name} {detail}")

    # --- official projection cubes -------------------------------------------------------
    cubes = {}
    for mig in MIG:
        years, N, D, B = read_projection(files[MIG_FILES[mig]])
        cubes[mig] = (N, D, B)
        net = (N[..., 1:, :, :].sum((-1, -2)) - N[..., :-1, :, :].sum((-1, -2))) - B[..., 1:, :].sum(-1) + D[..., 1:, :, :].sum((-1, -2))
        check(f"total accounting net={mig}", np.abs(net - mig).max() < 0.01, max_abs_error=float(np.abs(net - mig).max()))
    years = np.array(years)  # 2025..2080; 2025 is the first projected year-end
    Y = len(years)
    off_N = np.stack([cubes[m][0] for m in MIG])  # [mig, f, m, Y, 2, A]
    off_B = np.stack([cubes[m][2].sum(-1) for m in MIG])  # [mig, f, m, Y]
    off_D = np.stack([cubes[m][1].sum((-1, -2)) for m in MIG])

    # --- mortality: q[m, t, s, a] for t = 2026.. (exposure = aged population before deaths) ---
    N4, D4, B4 = cubes[400]
    aged = age_forward(N4[:, :, :-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        qall = np.where(aged > 0, D4[:, :, 1:] / aged, 0.0)  # [f, m, Y-1, 2, A]
    qmort = qall[CODES.index("MD")]  # [m, Y-1, 2, A]
    check("death probabilities in [0,1]", (qmort >= 0).all() and (qmort <= 1).all(), max_q=float(qmort.max()))
    check("mortality independent of fertility code", np.abs(qall - qmort[None]).max() < 1e-3, max_abs=float(np.abs(qall - qmort[None]).max()))

    # --- migration: M = M0 + net * P, exactly, for every assumption combination --------------
    R = {mig: migration_residual(*cubes[mig]) for mig in MIG}
    M0, P = R[0], (R[400] - R[0]) / 400.0
    lin = max(float(np.abs(M0 + mig * P - R[mig]).max()) for mig in (200, 600, 800))
    check("migration linear in net (held-out +200/+600/+800)", lin < 1e-3, max_abs_error_people=lin)
    check("per-person migration profile sums to 1", np.abs(P.sum((-1, -2)) - 1).max() < 1e-4, max_abs=float(np.abs(P.sum((-1, -2)) - 1).max()))

    # --- fertility: smooth age schedule x yearly intensity per fertility code ---------------------
    post_F = {}
    births = {}
    for mig in MIG:
        N_, D_, B_ = cubes[mig]
        post = age_forward(N_[:, :, :-1]) - D_[:, :, 1:]
        post_F[mig] = post[..., 0, :]  # [f, m, Y-1, A]
        births[mig] = B_[:, :, 1:].sum(-1)  # [f, m, Y-1]
    w = slice(0, 15)  # check window 2026-2040
    md = CODES.index("MD")
    table = next(d for d in manifest["documents"] if d["id"] == "S34")["table"]
    shape = published_shape(table)
    keys = [(mig, m) for mig in FERT_TRAIN for m in (0, md, 4)]
    intensity = np.zeros((5, Y - 1))
    for f in range(5):
        intensity[f] = _intensity(shape, {k: post_F[k[0]][f, k[1]] for k in keys}, {k: births[k[0]][f, k[1]] for k in keys}, keys)
    held = {}
    for mig in (200, 600):
        err = np.abs(intensity[:, None, :] * (post_F[mig] @ shape) - births[mig])[..., w]
        held[f"+{mig}"] = float(err.max())
    tfr_base = float(shape.sum())  # band rate is applied at each single age
    tfr_2034 = tfr_base * float(intensity[md][list(years[1:]).index(2034)])
    check("fertility shape predicts held-out scenario births", max(held.values()) < 5.0, max_abs_births_per_year=held, window="2026-2040")
    q["fertility"] = {
        "shape": "published Jersey age-specific fertility rates 2023-25 (S34 Table 2, p.11), uniform within 5-year bands 15-44",
        "tfr_of_shape": tfr_base,
        "intensity": "yearly scale factor on the 2023-25 rates, calibrated to official births for each fertility code",
        "calibrated_on": f"official net {FERT_TRAIN} x life-expectancy L2/MD/H2 cubes",
        "held_out_max_abs_births_error_2026_2040": held,
        "implied_tfr_2034_MD": tfr_2034, "published_tfr_2034_MD": 1.01,
        "note": "The published 2034 TFR was not used in calibration; agreement is an independent consistency check.",
    }
    sex_ratio = float((B4[..., 1] / B4[..., 0]).mean())
    check("birth sex ratio constant", float((B4[..., 1] / B4[..., 0]).std()) < 1e-3, ratio_m_per_f=sex_ratio)

    # --- observed estimates & baseline ---------------------------------------------------------
    obs_years, obs = read_estimates(files["S06"])
    check("estimates cover 2011-2025", obs_years[0] <= 2011 and obs_years[-1] >= 2025, years=[obs_years[0], obs_years[-1]])
    imputed = []

    def complete(year):
        x = obs[obs_years.index(year)].copy()
        ref = off_N[MIG.index(400), md, md, 0]  # official 2025 mid-range state
        for s, a in zip(*np.where(np.isnan(x))):
            x[s, a] = round(float(ref[s, a]))
            imputed.append({"year": year, "sex": SEX[s], "age": int(a), "value": x[s, a], "method": "official 2025 mid-range projected count, rounded"})
        return x

    base_2025 = complete(2025)
    base_2024 = complete(2024)
    q["baseline"] = {
        "year": 2025, "published_cells_total": float(np.nansum(obs[obs_years.index(2025)])),
        "imputed_suppressed_cells": [i for i in imputed if i["year"] == 2025],
        "total_after_imputation": float(base_2025.sum()),
        "official_projection_2025_mid": float(off_N[MIG.index(400), md, md, 0].sum()),
        "note": "Suppressed cells are not recovered; they are filled with a labelled modelling assumption.",
    }

    # --- LTC claim rates (2024 claims / 2024 population, same vintage) ----------------------------
    claims, ltc_status = read_ltc(files["S08"])
    check("2024 LTC claims for 65+ bands are published (not suppressed)", not np.isnan(claims).any(), status=ltc_status)
    denom = np.array([[base_2024[s, lo:hi + 1].sum() for s in range(2)] for _, lo, hi in LTC_BANDS])
    ltc_rate = claims / denom
    q["ltc"] = {"claims_2024": claims.tolist(), "population_2024": denom.tolist(), "rate": ltc_rate.tolist(), "bands": [b[0] for b in LTC_BANDS]}

    # --- observed components of change (retrospective checks only) -------------------------------
    comp = {int(r["Year"]): r for r in csv.DictReader(open(files["S35"], newline="", encoding="utf-8-sig"))}
    comp_years = [y for y in sorted(comp) if comp[y]["Births"] and comp[y]["Net migration"]]
    comp_arr = np.array([[float(comp[y][k] or 0) for k in ("Births", "Deaths", "Net migration", "Administrative adjustment", "End of year population")] for y in comp_years])
    check("components of change available 2012-2025", comp_years[0] <= 2012 and comp_years[-1] >= 2025, years=[comp_years[0], comp_years[-1]])

    # --- school participation (January census Y vs population at end of Y-1) ----------------------
    edu = {int(r["Year"]): r for r in csv.DictReader(open(files["S36"], newline="", encoding="utf-8-sig"))}
    ratios = {}
    for y in sorted(edu):
        if y - 1 in obs_years:
            pop = np.nan_to_num(obs[obs_years.index(y - 1)])
            prim = float(edu[y]["Government primary"]) + float(edu[y]["Non-provided primary"])
            sec = float(edu[y]["Government secondary"]) + float(edu[y]["Non-provided secondary"])
            ratios[y] = (prim / float((pop * W_PRIMARY).sum()), sec / float((pop * W_SECONDARY).sum()), prim, sec)
    last = max(ratios)
    school_ratio = np.array(ratios[last][:2])
    rp = [v[0] for v in ratios.values()]
    rs = [v[1] for v in ratios.values()]
    check("school participation ratios computed", len(ratios) >= 10, years=[min(ratios), last])
    q["schools"] = {
        "census_year_used": last, "primary_ratio": school_ratio[0], "secondary_ratio": school_ratio[1],
        "primary_ratio_range": [min(rp), max(rp)], "secondary_ratio_range": [min(rs), max(rs)],
        "by_year": {y: {"primary_ratio": round(v[0], 4), "secondary_ratio": round(v[1], 4), "primary_pupils": v[2], "secondary_pupils": v[3]} for y, v in ratios.items()},
        "definition": "January pupils (government + non-provided; special schools excluded) / residents of matching school age at the previous 31 December (year-group weights by month of birth).",
    }

    # --- health service use weights (S37) and workforce baseline (S38) -------------------------
    t37 = next(d for d in manifest["documents"] if d["id"] == "S37")["table"]
    w_bed, w_gp = np.zeros((2, A)), np.zeros((2, A))
    for i, band in enumerate(t37["bands"]):
        lo, hi = (90, 100) if band == "90+" else map(int, band.split("-"))
        for s_, sx in enumerate("FM"):
            w_bed[s_, lo:hi + 1] = t37[f"bed_days_{sx}"][i]
            w_gp[s_, lo:hi + 1] = t37[f"gp_appointments_{sx}"][i]
    p23 = np.nan_to_num(obs[obs_years.index(2023)])
    bed23, gp23 = float((p23 * w_bed).sum()), float((p23 * w_gp).sum())
    check("bed-day weights reproduce PHI 2023 total (60,300) within 2%", abs(bed23 / 60300 - 1) < 0.02, ours=round(bed23))
    check("GP weights reproduce PHI 2021-23 total (410,790) within 2%", abs(gp23 / 410790 - 1) < 0.02, ours=round(gp23))
    t38 = next(d for d in manifest["documents"] if d["id"] == "S38")["table"]
    staff = np.array([t38["nurses_midwives_fte"], t38["doctors_fte"]], dtype=float)
    q["health"] = {"bed_days_2023_ours": bed23, "bed_days_2023_phi": 60300, "gp_2023_ours": gp23, "gp_2021_23_phi": 410790,
                   "staff_baseline_fte_2025_12_31": {"nurses_midwives": staff[0], "doctors": staff[1]},
                   "method": "Hospital FTE held constant per age-weighted bed day (PHI Fig. 39 rates, 2018-22); baseline Health and Care Jersey FTE at 31 Dec 2025."}

    # --- homes needed (official housing-needs method), exactly linear in net migration ---------------
    hrows = list(csv.DictReader(open(files["S39"], newline="", encoding="utf-8-sig")))
    htypes = sorted({r["Housing type"] for r in hrows}, key=lambda t: (t.split(",")[0].split()[0], t))
    hyears = sorted({int(r["Year"]) for r in hrows})
    hnet = {"Net nil": 0, "+200": 200, "+400": 400, "+600": 600, "+800": 800}
    H = np.zeros((5, len(hyears), len(htypes)))
    for r in hrows:
        H[MIG.index(hnet[r["Net migration"].strip()]), hyears.index(int(r["Year"])), htypes.index(r["Housing type"])] += float(r["Households"])
    h_nil, h_per = H[0], (H[2] - H[0]) / 400.0
    hlin = max(float(np.abs(h_nil + mig * h_per - H[MIG.index(mig)]).max()) for mig in (200, 600, 800))
    check("homes needed linear in net migration (held-out +200/+600/+800)", hlin < 1e-2, max_abs_error_homes=hlin)
    q["housing"] = {"years": [hyears[0], hyears[-1]], "types": htypes, "additional_2040_by_scenario": {str(mig): round(float((H[i, -1] - H[i, 0]).sum()), 1) for i, mig in enumerate(MIG)},
                    "per_100_net_migration_2040": round(float(100 * (h_per[-1] - h_per[0]).sum()), 1),
                    "note": "Emulates the official housing-needs projections (headship rates x private-household population, one dwelling per household, 2021 Census dwelling-type ratios). Mid-range fertility and life expectancy assumed by the source."}

    # --- write -------------------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        years=years, off_N=off_N, off_B=off_B, off_D=off_D, q=qmort, M0=M0, P=P,
        fert_shape=shape, fert_intensity=intensity, sex_ratio=np.array(sex_ratio),
        obs_years=np.array(obs_years), obs=obs, base_2025=base_2025,
        ltc_rate=ltc_rate, ltc_claims=claims, comp_years=np.array(comp_years), comp=comp_arr,
        w_primary=W_PRIMARY, w_secondary=W_SECONDARY, school_ratio=school_ratio,
        w_bed=w_bed, w_gp=w_gp, staff_fte=staff, house_years=np.array(hyears), house_nil=h_nil, house_per=h_per, house_types=np.array(htypes),
        edu_years=np.array(sorted(ratios)), edu_pupils=np.array([ratios[y][2:] for y in sorted(ratios)]),
    )
    np.savez_compressed(out / "pack.npz", **arrays)
    src_hash = hashlib.sha256("".join(sorted([s["sha256"] or sha256(files[s["id"]]) for s in manifest["sources"]] + [d["sha256"] for d in manifest["documents"] if d.get("sha256")] + [sha256(__file__)])).encode()).hexdigest()
    pack_id = f"{PACK_VERSION}-{src_hash[:12]}"
    meta = {
        "pack_id": pack_id, "pack_version": PACK_VERSION, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [{k: s.get(k) for k in ("id", "title", "resource_id", "url", "vintage", "licence", "sha256", "bytes", "use")} for s in manifest["sources"]],
        "codes": CODES, "migration_scenarios": MIG, "ltc_bands": [b[0] for b in LTC_BANDS],
        "years": [int(years[0]), int(years[-1])], "gate": "B", "quality": q,
    }
    (out / "pack.json").write_text(json.dumps(meta, indent=2))
    (out / "data_quality.json").write_text(json.dumps(q, indent=2))
    print(f"pack {pack_id}: {len(q['checks'])} checks passed; TFR2034={tfr_2034:.3f}; held-out births err {held}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/processed", type=Path)
    ap.add_argument("--manifest", default="data/manifests/sources.json", type=Path)
    ap.add_argument("--pin", action="store_true", help="record sha256/bytes of downloaded files into the manifest")
    a = ap.parse_args()
    build(a.raw, a.out, a.pin, a.manifest)
