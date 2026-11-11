"""nomad2018 결정 구조(geometry.xyz) 피처: MLE-bench 에이전트 Codex × GPT 풀이에서 옮김 (C1 의 F2, docs/plans/roadmap.md).

출처: 실행 nomad2018-predict-transparent-conductors__codex-gpt__r1__20261109-200654 (test 0.04847) 의 작업 디렉터리.
에이전트 코드를 거의 그대로 옮기고 경로·병렬·캐시만 바꿈 (계산식·상수·순서는 같음 → 에이전트가 만든 배열과 같은 값):
  lattice   phase_kfold.py:11-17   원자당 부피, 각도의 90°·120° 거리, 격자 길이 비 (10열)
  symmetry  symmetry.py            양이온 자리를 사면체/팔면체로 나눈 O 거리 통계 (16열, sg 227·194 만 값, 나머지 0)
  radial    geometry_features.py   원소 쌍 16개 × (거리 가우시안 bin 39개 + 자리별 이웃 거리·배위수 통계 28개) (1,072열)
  soap      geometry_features.py   SOAP (r_cut 5, n_max 6, l_max 4, sigma 0.45) 원소별 평균·표준편차 (6,000열 × 2)
  topology  topology.py            자리 종류 12개 × 이웃 수·이웃 거리·O 주변 양이온 조성 (14,071열, sg 227·194 만)
  fourier   fourier.py             자리 종류 10개의 구조 인자 교차항 (8,580열, sg 227·194 만)
  spectrum  spectrum.py            양이온 사이 거리 가중 행렬(밀착 결합 꼴)의 고유값 통계 (3,072열, sg 227 만)
빼 둔 것 (타깃을 쓰거나 피처가 아닌 것): 정답 섞기(finalize_submission.py:19-28, train 과 같은 구조인 test 행에 train 정답을
섞음), fold 마다 타깃 상관으로 열 고르기(phase_kfold.py:23-26), 복셀·그래프·원자별 SOAP (CNN·GNN·커널 모델의 입력).

행 순서는 train 다음 test (build 가 넘겨받은 순서). 무거운 패키지(ase, dscribe)는 함수 안에서 import → 로컬에서 변형 파일을
읽어 PIP 만 볼 때는 필요 없음. 블록마다 data_dir/_features/ 에 캐시 (한 작업 안에서 모델마다 새 프로세스로 돌 때 다시 안 만듦).
"""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PIP = [["ase==3.29.0", "dscribe==2.1.2"]]  # 에이전트는 버전 없이 설치 (2026-11-09). 2026-11-11 최신으로 고정
CACHE_VERSION = 1  # 계산을 바꾸면 올림 (예전 캐시를 안 씀)
SPECIES = ["Al", "Ga", "In", "O"]
TARGETS = ["formation_energy_ev_natom", "bandgap_energy_ev"]


def read_xyz(path: Path):
    lattice, coords, symbols = [], [], []
    with open(path) as f:
        for line in f:
            v = line.split()
            if not v:
                continue
            if v[0] == "lattice_vector":
                lattice.append([float(x) for x in v[1:4]])
            if v[0] in ("atom", "atom_frac"):
                coords.append([float(x) for x in v[1:4]])
                symbols.append(v[4])
                if v[0] == "atom_frac":
                    coords[-1] = np.asarray(coords[-1]) @ np.asarray(lattice)
    return lattice, coords, symbols


def atoms_of(path: Path):
    from ase import Atoms

    lattice, coords, symbols = read_xyz(path)
    return Atoms(symbols=symbols, positions=coords, cell=lattice, pbc=True), np.array(symbols), np.array(lattice)


def is_tet(i, j, d, z, site):
    """양이온 site 가 사면체 자리인지: 가까운 O 다섯째 거리가 넷째의 1.18 배보다 멀면 (에이전트 기준 그대로)."""
    ds = np.sort(d[(i == site) & (z[j] == "O")])
    return len(ds) > 4 and ds[4] > ds[3] * 1.18


# ---------- 블록 (구조 하나 → 1차원 배열, 해당 없는 공간군은 None = 0 으로 채움) ----------

def radial_soap(path: Path, sg: int):
    from ase.neighborlist import neighbor_list

    soap = _soap()
    atoms, z, _ = atoms_of(path)
    s = soap.create(atoms)
    s = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    averages, stdev = [], []
    for elem in SPECIES:
        sel = s[z == elem]
        if len(sel):
            averages.append(sel.sum(axis=0) / len(z))
            stdev.append(sel.std(axis=0) * len(sel) / len(z))
        else:
            averages.append(np.zeros(s.shape[1], np.float32))
            stdev.append(np.zeros(s.shape[1], np.float32))
    i, j, d = neighbor_list("ijd", atoms, 5.0)
    centers = np.arange(1.2, 5.01, .1)
    feats = []
    for a in SPECIES:
        for b in SPECIES:
            mask = (z[i] == a) & (z[j] == b)
            dd = d[mask]
            hist = np.exp(-.5 * ((dd[:, None] - centers) / .12) ** 2).sum(axis=0) / len(z) if len(dd) \
                else np.zeros(len(centers))
            feats.extend(hist)
            inds = i[mask]
            sitevals = []
            for site in np.where(z == a)[0]:
                ds = np.sort(dd[inds == site])
                if len(ds):
                    sitevals.append([ds[0], np.mean(ds[:min(4, len(ds))]), np.sum(ds < 2.1), np.sum(ds < 2.3),
                                     np.sum(ds < 2.5), np.sum(ds < 2.8), np.sum(ds < 3.2)])
            if sitevals:
                sitevals = np.array(sitevals)
                for stat in (np.mean, np.std, np.min, np.max):
                    feats.extend(stat(sitevals, axis=0))
            else:
                feats.extend(np.zeros(28))
    return dict(radial=np.array(feats, np.float32), soap_mean=np.concatenate(averages),
                soap_std=np.concatenate(stdev))


_SOAP = None


def _soap():
    global _SOAP
    if _SOAP is None:
        from dscribe.descriptors import SOAP

        _SOAP = SOAP(species=SPECIES, periodic=True, r_cut=5, n_max=6, l_max=4, sigma=0.45, dtype="float32")
    return _SOAP


def symmetry(path: Path, sg: int):
    from ase.neighborlist import neighbor_list

    if sg not in (227, 194):
        return None
    atoms, z, _ = atoms_of(path)
    i, j, d = neighbor_list("ijd", atoms, 4)
    tet, octa = [], []
    for t in np.where(z != "O")[0]:
        d0 = np.sort(d[(i == t) & (z[j] == "O")])
        if len(d0) < 6:
            d0 = np.pad(d0, (0, 6 - len(d0)), constant_values=4)
        (tet if d0[4] > d0[3] * 1.18 else octa).append(d0[:6])
    result = []
    for typ, x in enumerate([tet, octa]):
        if len(x):
            x = np.array(x)
            v = x[:, :4] if typ == 0 else x[:, :6]
            result.extend([v.mean(), v.std(), v.mean(axis=1).std(), v.std(axis=1).mean(), v.std(axis=1).std(),
                           np.ptp(v.mean(axis=1)), v.max() - v.min(), len(v) / len(z)])
        else:
            result.extend([0] * 8)
    return np.array(result, np.float64)


_COMPS = {4: [(a, b, 4 - a - b) for a in range(5) for b in range(5 - a)],
          6: [(a, b, 6 - a - b) for a in range(7) for b in range(7 - a)]}


def topology(path: Path, sg: int):
    from ase.neighborlist import neighbor_list

    if sg not in (227, 194):
        return None
    atoms, z, _ = atoms_of(path)
    n = len(z)
    ii, jj, dd = neighbor_list("ijd", atoms, 6.0)
    iss = np.arange(n)
    tet, octa = np.zeros(n, bool), np.zeros(n, bool)
    for i in iss[z != "O"]:
        tet[i] = is_tet(ii, jj, dd, z, i)
        octa[i] = not tet[i]
    groups = [z == "Al", z == "Ga", z == "In", z == "O", (z == "Al") & tet, (z == "Ga") & tet, (z == "In") & tet,
              (z == "Al") & octa, (z == "Ga") & octa, (z == "In") & octa, tet, octa]
    features = []
    for mask in groups:
        inds = iss[mask]
        features.append(len(inds) / n)
        for nbmask in groups[:10]:
            selected = mask[ii] & nbmask[jj]
            ind, dist = ii[selected], dd[selected]
            for cut in [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.01]:
                count = np.bincount(ind[dist < cut], minlength=n)[inds]
                if len(count):
                    features.extend([count.mean(), count.std(), count.min(), count.max()])
                    features.extend([np.mean(count == k) for k in range(9)])
                else:
                    features.extend([0] * 13)
        for nb in [z == "O", z == "Al", z == "Ga", z == "In"]:
            selected = mask[ii] & nb[jj]
            ind, dist = ii[selected], dd[selected]
            ds = []
            for i in inds:
                u = np.sort(dist[ind == i])
                ds.append(np.pad(u[:8], (0, max(8 - len(u), 0)), constant_values=6))
            if len(ds):
                ds = np.array(ds)
                for stat in (np.mean, np.std, np.min, np.max):
                    features.extend(stat(ds, axis=0))
            else:
                features.extend([0] * 32)
    oxy = iss[z == "O"]
    for k, compositions in _COMPS.items():
        counts = []
        for i in oxy:
            sel = (ii == i) & (z[jj] != "O")
            order = np.argsort(dd[sel])
            syms = z[jj[sel]][order[:k]]
            counts.append(tuple(np.sum(syms == e) for e in ["Al", "Ga", "In"]))
        for c in compositions:
            features.append(sum(x == c for x in counts) / max(len(oxy), 1))
    return np.array(features, np.float32)


_HKS = [(h, k) for h in range(-3, 4) for k in range(-3, 4) if h * h + k * k + h * k <= 9]
_SHELLS = sorted(set(h * h + k * k + h * k for h, k in _HKS))
_PAIRS = np.triu_indices(10)


def fourier(path: Path, sg: int):
    from ase.neighborlist import neighbor_list

    if sg not in (227, 194):
        return None
    atoms, z, lattice = atoms_of(path)
    coords = atoms.get_positions()
    i, j, d = neighbor_list("ijd", atoms, 4)
    tet = np.zeros(len(z), bool)
    for t in np.where(z != "O")[0]:
        tet[t] = is_tet(i, j, d, z, t)
    octa = (z != "O") & (~tet)
    a, b, c = lattice  # 30° 육방 격자를 표준 120° 격자로 (에이전트 주석)
    gamma = np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)) * 180 / np.pi
    if gamma < 50:
        a, b = a - b, -b
        lattice = np.array([a, b, c])
    frac = coords @ np.linalg.inv(lattice)
    frac %= 1
    groups = [z == "Al", z == "Ga", z == "In", z == "O", (z == "Al") & tet, (z == "Ga") & tet, (z == "In") & tet,
              (z == "Al") & octa, (z == "Ga") & octa, (z == "In") & octa]
    group = np.array(groups, dtype=float)
    hkl = np.array([(h, k, m) for m in range(0, 13) for h, k in _HKS])
    amps = np.exp(2j * np.pi * (hkl @ frac.T)) @ group.T / len(z)
    out = []
    for m in range(13):
        for shell in _SHELLS:
            mask = (hkl[:, 2] == m) & ((hkl[:, 0] ** 2 + hkl[:, 1] ** 2 + hkl[:, 0] * hkl[:, 1]) == shell)
            F = amps[mask]
            cs = (F[:, :, None] * F[:, None, :].conj())[:, _PAIRS[0], _PAIRS[1]]
            out.extend(cs.real.mean(axis=0))
            out.extend(np.abs(cs.imag.mean(axis=0)))
    return np.array(out, np.float32)


_KPOINTS = np.array(list(product([0, .5], repeat=3)))


def spectrum(path: Path, sg: int):
    from ase.neighborlist import neighbor_list

    if sg != 227:
        return None
    atoms, z, _ = atoms_of(path)
    inds = np.where(z != "O")[0]
    mapping = np.zeros(len(z), int)
    mapping[inds] = np.arange(len(inds))
    n = len(inds)
    i, j, d, shift = neighbor_list("ijdS", atoms, 7)
    tet = np.array([is_tet(i, j, d, z, atom) for atom in inds], bool)
    keep = (z[i] != "O") & (z[j] != "O")
    ii, jj, dist, delta = mapping[i[keep]], mapping[j[keep]], d[keep], shift[keep]
    onsite = np.array([{"Al": 3., "Ga": 1.4, "In": 0}[x] for x in z[inds]])
    features = []
    for r in [3.3, 4, 5, 6.5]:
        for alpha in [.5, 1.5]:
            weight = np.exp(-alpha * (dist - 3)) * np.exp(-((dist / r) ** 6))
            phases = np.exp(2j * np.pi * (_KPOINTS @ delta.T))
            matrices = []
            for phase in phases:
                A = np.zeros((n, n), dtype=complex)
                np.add.at(A, (ii, jj), weight * phase)
                matrices.append((A + A.conj().T) / 2)
            A = np.array(matrices)
            for shift_tet in [0, 1.0]:
                diag = onsite + shift_tet * tet
                for hop in [.3, .8, 1.5]:
                    H = hop * A.copy()
                    H[:, np.arange(n), np.arange(n)] += diag
                    eig = np.linalg.eigvalsh(H)
                    for stat in (np.mean, np.min, np.max, np.std):
                        features.extend(stat(eig, axis=0))
    return np.array(features, np.float32)


BLOCKS = {"symmetry": symmetry, "radial_soap": radial_soap, "topology": topology, "fourier": fourier,
          "spectrum": spectrum}


# ---------- 조립 ----------

def _items(train: pd.DataFrame, test: pd.DataFrame, data_dir: Path):
    return [(data_dir / g / str(int(r.id)) / "geometry.xyz", int(r.spacegroup))
            for g, df in (("train", train), ("test", test)) for r in df.itertuples()]


def _stack(values: list) -> np.ndarray:
    """구조마다의 배열 → 행렬. None(해당 없는 공간군)은 0. 길이가 다르면 (spectrum 은 양이온 수에 따라 다를 수 있음,
    에이전트 데이터에서는 모두 같았음) 가장 긴 길이에 0 을 채움."""
    width = max((len(v) for v in values if v is not None), default=0)
    out = np.zeros((len(values), width), np.float32)
    for k, v in enumerate(values):
        if v is not None:
            out[k, :len(v)] = v
    return out


def block(name: str, train: pd.DataFrame, test: pd.DataFrame, data_dir: Path) -> dict[str, np.ndarray]:
    """블록 하나를 train+test 전체에 대해 계산 (캐시). {열 이름 접두어: 행렬}."""
    from joblib import Parallel, delayed

    ids = np.r_[train.id.to_numpy(), test.id.to_numpy()]
    cache = Path(data_dir) / "_features" / f"{name}_v{CACHE_VERSION}.npz"
    if cache.exists():  # 다른 모델 프로세스가 만든 것. 읽지 못하면 (깨짐 등) 다시 계산
        try:
            with np.load(cache) as z:
                if np.array_equal(z["_ids"], ids):
                    return {k: z[k] for k in z.files if k != "_ids"}
        except (OSError, ValueError, KeyError) as e:
            print(f"[features] 캐시 읽기 실패 → 다시 계산: {e!r}")
    fn = BLOCKS[name]
    n_jobs = int(os.environ.get("FAB_THREADS") or os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1)
    res = Parallel(n_jobs=n_jobs)(delayed(fn)(p, sg) for p, sg in _items(train, test, Path(data_dir)))
    if name == "radial_soap":
        out = {k: _stack([r[k] for r in res]) for k in ("radial", "soap_mean", "soap_std")}
    else:
        out = {name: _stack(res)}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")  # 임시 파일 → 교체: 동시에 도는 프로세스가 반쯤 쓴 파일을 안 읽게
        np.savez(tmp, _ids=ids, **out)
        os.replace(tmp, cache)
    except OSError as e:  # 캐시는 있으면 좋은 것
        print(f"[features] 캐시 저장 실패 (무시): {e!r}")
    return out


def lattice(rows: pd.DataFrame) -> pd.DataFrame:
    """에이전트 phase_kfold.py 의 표 파생 열 (원본 열은 기본 피처가 이미 가짐)."""
    out = pd.DataFrame(index=rows.index)
    angles = ["lattice_angle_alpha_degree", "lattice_angle_beta_degree", "lattice_angle_gamma_degree"]
    lengths = ["lattice_vector_1_ang", "lattice_vector_2_ang", "lattice_vector_3_ang"]
    cs = np.cos(np.deg2rad(rows[angles].to_numpy(float)))
    vol = np.prod(rows[lengths].to_numpy(float), axis=1) * np.sqrt(
        np.maximum(0, 1 + 2 * np.prod(cs, axis=1) - (cs ** 2).sum(axis=1)))
    out["volume_atom"] = vol / rows.number_of_total_atoms.to_numpy(float)
    for a in angles:
        out[a + "_dist90"] = np.abs(rows[a] - 90)
        out[a + "_dist120"] = np.abs(rows[a] - 120)
    for a in [1, 2, 3]:
        for b in [1, 2, 3]:
            if a < b:
                out[f"length_ratio_{a}{b}"] = rows[f"lattice_vector_{a}_ang"] / rows[f"lattice_vector_{b}_ang"]
    return out


def build(train: pd.DataFrame, test: pd.DataFrame, data_dir: Path, blocks: list[str]):
    """기본 피처(F0, 원본 열) + 격자 파생 + blocks 의 구조 피처. blocks 의 이름은 열 접두어 (radial·soap_mean 등)."""
    base = _base()
    X, X_test, cat_cols = base(train, test)
    rows = pd.concat([train.drop(columns=[c for c in TARGETS if c in train]), test], ignore_index=True)
    parts = [lattice(rows)]
    need = {"radial": "radial_soap", "soap_mean": "radial_soap", "soap_std": "radial_soap"}
    computed: dict[str, np.ndarray] = {}
    for b in blocks:
        if b not in computed:
            computed |= block(need.get(b, b), train, test, data_dir)
        parts.append(pd.DataFrame(computed[b], columns=[f"{b}_{k}" for k in range(computed[b].shape[1])]))
    extra = pd.concat(parts, axis=1)
    n = len(train)
    X = pd.concat([X.reset_index(drop=True), extra.iloc[:n].reset_index(drop=True)], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), extra.iloc[n:].reset_index(drop=True)], axis=1)
    return X, X_test, cat_cols


def _base():
    """대회의 기본 features.py 의 build (F0)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("features_mleb_nomad2018_formation_base",
                                                  Path(__file__).parents[1] / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build
