"""
src/permissions_model_1_6.py

Basic mode: resource allowed if they did the activity in the logs.
Advanced mode: implication mining + transitive closure expands permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import joblib
import pandas as pd
import numpy as np


def _load_perm_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    # some rows in BPI 2017 have columns collapsed into "Action"
    if {"Action", "concept:name", "org:resource"}.issubset(df.columns):
        broken = (
            (df["concept:name"].isna() | df["org:resource"].isna())
            & df["Action"].astype(str).str.contains(",", regex=False)
        )
        if broken.any():
            parts = df.loc[broken, "Action"].astype(str).str.split(",", n=18, expand=True)
            df.loc[broken & df["concept:name"].isna(), "concept:name"] = parts[0].values
            df.loc[broken & df["org:resource"].isna(), "org:resource"] = parts[1].values

    df = df[["concept:name", "org:resource"]].dropna().copy()
    df["concept:name"] = df["concept:name"].astype(str).str.strip()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()

    # Filter for valid BPI 2017 activities (A_*, O_*, W_*)
    df = df[df["concept:name"].str.match(r"^[AOW]_", na=False)].copy()
    return df


@dataclass(frozen=True)
class PermissionArtifactBasic:
    activity_to_resources: Dict[str, List[str]]


@dataclass(frozen=True)
class PermissionArtifactAdvanced:
    activity_to_resources_basic: Dict[str, List[str]]
    resource_base_perms: Dict[str, Set[str]]
    resource_expanded_perms: Dict[str, Set[str]]


def _tarjan_scc(nodes: List[str], graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Tarjan's SCC algorithm — needed to handle cycles in the implication graph."""
    index = 0
    stack: List[str] = []
    onstack: Set[str] = set()
    idx: Dict[str, int] = {}
    low: Dict[str, int] = {}
    sccs: List[List[str]] = []

    def strongconnect(v: str):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1

        stack.append(v)
        onstack.add(v)

        for w in graph.get(v, set()):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in nodes:
        if v not in idx:
            strongconnect(v)

    return sccs


def _train_basic(df_perm: pd.DataFrame) -> PermissionArtifactBasic:
    activity_to_resources = (
        df_perm.drop_duplicates()
        .groupby("concept:name", observed=True)["org:resource"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )
    return PermissionArtifactBasic(activity_to_resources=activity_to_resources)


def _train_advanced(
    df_perm: pd.DataFrame,
    *,
    min_support: int = 10,
    min_imp: float = 0.95,
) -> PermissionArtifactAdvanced:
    # 1. base mapping
    activity_to_resources_basic = (
        df_perm.drop_duplicates()
        .groupby("concept:name", observed=True)["org:resource"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )

    pairs = df_perm.drop_duplicates().copy()
    activity_sets: Dict[str, Set[str]] = (
        pairs.groupby("concept:name", observed=True)["org:resource"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )

    activities = list(activity_sets.keys())

    resource_base: DefaultDict[str, Set[str]] = defaultdict(set)
    for act, res_set in activity_sets.items():
        for r in res_set:
            resource_base[r].add(act)

    # 2. implication graph
    imp_graph: Dict[str, Set[str]] = {a: set() for a in activities}

    for B in activities:
        RB = activity_sets[B]
        suppB = len(RB)
        if suppB < min_support:
            continue
        for A in activities:
            if A == B:
                continue
            # fraction of B-doers who also do A
            inter = len(RB & activity_sets[A])
            imp = inter / suppB if suppB > 0 else 0.0
            if imp >= min_imp:
                imp_graph[B].add(A)

    # 3. collapse SCCs
    sccs = _tarjan_scc(activities, imp_graph)
    activity_to_group: Dict[str, int] = {}
    for gid, group in enumerate(sccs):
        for act in group:
            activity_to_group[act] = gid

    group_to_activities: Dict[int, Set[str]] = {gid: set(group) for gid, group in enumerate(sccs)}

    # build group DAG
    group_edges: DefaultDict[int, Set[int]] = defaultdict(set)
    for B, outs in imp_graph.items():
        gB = activity_to_group[B]
        for A in outs:
            gA = activity_to_group[A]
            if gA != gB:
                group_edges[gB].add(gA)

    # 4. reachability via DFS
    group_reach: Dict[int, Set[int]] = {}

    def dfs(g: int) -> Set[int]:
        if g in group_reach:
            return group_reach[g]
        reach = {g}
        for nxt in group_edges.get(g, set()):
            reach |= dfs(nxt)
        group_reach[g] = reach
        return reach

    for g in range(len(sccs)):
        dfs(g)

    # 5. expand permissions
    resource_expanded: Dict[str, Set[str]] = {}
    for r, base_acts in resource_base.items():
        base_groups = {activity_to_group[a] for a in base_acts}
        expanded_groups: Set[int] = set()
        for g in base_groups:
            expanded_groups |= group_reach[g]
        expanded_acts: Set[str] = set()
        for g in expanded_groups:
            expanded_acts |= group_to_activities[g]
        resource_expanded[r] = expanded_acts

    return PermissionArtifactAdvanced(
        activity_to_resources_basic=activity_to_resources_basic,
        resource_base_perms=dict(resource_base),
        resource_expanded_perms=resource_expanded,
    )


class PermissionsModel:

    def __init__(
        self,
        csv_path: str,
        mode: str = "basic",
        *,
        cache_path: Optional[str] = None,
        min_support: int = 10,
        min_imp: float = 0.95,
    ):
        self.csv_path = str(csv_path)
        self.mode = mode.lower().strip()
        if self.mode not in {"basic", "advanced"}:
            raise ValueError("mode must be 'basic' or 'advanced'")

        self.min_support = int(min_support)
        self.min_imp = float(min_imp)

        # default cache path
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[1]
            models_dir = project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(models_dir / f"permissions_model_1_6_{self.mode}.pkl")

        self.cache_path = cache_path
        self.artifact = self._load_or_train()

    def _load_or_train(self):
        p = Path(self.cache_path)
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass

        print(f"[PermissionsModel] Training new model (Mode: {self.mode})...")
        df_perm = _load_perm_df(self.csv_path)
        
        if self.mode == "basic":
            art = _train_basic(df_perm)
        else:
            art = _train_advanced(df_perm, min_support=self.min_support, min_imp=self.min_imp)

        try:
            joblib.dump(art, p)
            print(f"[PermissionsModel] Saved artifact to {p}")
        except Exception:
            pass

        return art

    def get_allowed_resources(self, activity: str) -> List[str]:
        activity = str(activity)
        if self.mode == "basic":
            return list(self.artifact.activity_to_resources.get(activity, []))

        # advanced: reverse-map expanded permissions
        out: List[str] = []
        for r, perms in self.artifact.resource_expanded_perms.items():
            if activity in perms:
                out.append(r)
        return out

    def can_execute(self, resource: str, activity: str) -> bool:
        resource = str(resource)
        activity = str(activity)
        
        if self.mode == "basic":
            return resource in set(self.artifact.activity_to_resources.get(activity, []))
            
        return activity in self.artifact.resource_expanded_perms.get(resource, set())