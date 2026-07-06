"""
Tile Grid Computation

Author: Valerie Foley
Last Updated: 5/2026

Description:
    Computes the grid of sub-tiles for tiled cluster processing. Each
    sub-tile has a core_bbox (the region it owns in the final mosaic;
    adjacent cores tile perfectly with no gap or overlap), a full_bbox
    (core + overlap on each side; the extent passed to FlowPy, overlapping
    neighbors by overlap_m per shared edge), and is_edge (True on the
    cluster boundary). Guarantees: the union of cores covers the cluster
    bbox exactly, adjacent full_bboxes overlap by exactly overlap_m, and
    tile IDs are stable across runs (same inputs -> same IDs -> same files).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Tile:
    # Single sub-tile in a cluster's tile grid.
    tile_id: str                      # e.g. "00_03"
    row: int                          # row index (0 at south, increases north)
    col: int                          # column index (0 at west, increases east)
    core_bbox: Tuple[float, float, float, float]   # (minx, miny, maxx, maxy)
    full_bbox: Tuple[float, float, float, float]   # core + overlap
    is_edge: bool                     # True if any side has no neighbor

    def core_size_m(self) -> Tuple[float, float]:
        # @returns: (width, height) of core_bbox in meters
        return (self.core_bbox[2] - self.core_bbox[0],
                self.core_bbox[3] - self.core_bbox[1])

    def full_size_m(self) -> Tuple[float, float]:
        # @returns: (width, height) of full_bbox in meters
        return (self.full_bbox[2] - self.full_bbox[0],
                self.full_bbox[3] - self.full_bbox[1])

    def core_centroid(self) -> Tuple[float, float]:
        # Center of core_bbox, used for centroid-distance ownership of
        # release polygons.
        # @returns: (x, y)
        cx = (self.core_bbox[0] + self.core_bbox[2]) / 2.0
        cy = (self.core_bbox[1] + self.core_bbox[3]) / 2.0
        return cx, cy


def compute_tile_grid(cluster_bbox: Tuple[float, float, float, float],
                       core_m: float = 3000.0,
                       overlap_m: float = 3000.0) -> List[Tile]:
    # Divide cluster_bbox into a grid of core regions, each surrounded by
    # an overlap buffer.
    # @param cluster_bbox: (minx, miny, maxx, maxy) in projected meters
    # @param core_m: edge length of each tile's core (region it owns)
    # @param overlap_m: overlap added to each side of the core -> full extent
    # @returns: list of Tile, sorted by (row, col)

    minx, miny, maxx, maxy = cluster_bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError(f"Invalid cluster_bbox: {cluster_bbox}")
    if core_m <= 0 or overlap_m < 0:
        raise ValueError(f"core_m must be positive, overlap_m non-negative")

    width = maxx - minx
    height = maxy - miny

    # ceil so the last core extends past the cluster edge if it doesn't
    # divide evenly; the overhang is clipped to AOI downstream anyway.
    n_cols = max(1, math.ceil(width / core_m))
    n_rows = max(1, math.ceil(height / core_m))

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            core_minx = minx + col * core_m
            core_miny = miny + row * core_m
            core_maxx = core_minx + core_m
            core_maxy = core_miny + core_m

            full_minx = core_minx - overlap_m
            full_miny = core_miny - overlap_m
            full_maxx = core_maxx + overlap_m
            full_maxy = core_maxy + overlap_m

            is_edge = (row == 0 or row == n_rows - 1 or
                       col == 0 or col == n_cols - 1)

            tiles.append(Tile(
                tile_id=f"{row:02d}_{col:02d}",
                row=row,
                col=col,
                core_bbox=(core_minx, core_miny, core_maxx, core_maxy),
                full_bbox=(full_minx, full_miny, full_maxx, full_maxy),
                is_edge=is_edge,
            ))

    return tiles


# --------- Smoke Test / CLI ---------

def _summarize_grid(tiles: List[Tile], cluster_bbox: Tuple) -> None:
    # Print a human-readable grid summary for debugging.
    if not tiles:
        print("(no tiles)")
        return
    n_rows = max(t.row for t in tiles) + 1
    n_cols = max(t.col for t in tiles) + 1
    cluster_w = cluster_bbox[2] - cluster_bbox[0]
    cluster_h = cluster_bbox[3] - cluster_bbox[1]

    print(f"Cluster bbox: {cluster_bbox}")
    print(f"             {cluster_w/1000:.1f} km x {cluster_h/1000:.1f} km")
    print(f"Grid:         {n_rows} rows x {n_cols} cols  ({len(tiles)} tiles)")
    if tiles:
        print(f"Core size:   {tiles[0].core_size_m()[0]/1000:.1f} km x "
              f"{tiles[0].core_size_m()[1]/1000:.1f} km")
        print(f"Full size:   {tiles[0].full_size_m()[0]/1000:.1f} km x "
              f"{tiles[0].full_size_m()[1]/1000:.1f} km")

    # Total core area should be >= cluster area (cores extend past edge)
    total_core_area = sum(
        (t.core_bbox[2] - t.core_bbox[0]) *
        (t.core_bbox[3] - t.core_bbox[1])
        for t in tiles
    )
    cluster_area = cluster_w * cluster_h
    print(f"Core coverage: {total_core_area/cluster_area*100:.1f}% of cluster "
          f"(>=100% expected; extends past edge)")

    n_edge = sum(1 for t in tiles if t.is_edge)
    print(f"Edge tiles:   {n_edge}/{len(tiles)}")


if __name__ == "__main__":
    test_cases = [
        ("cluster_11 (4.3 x 4.3 km, smallest)",  (0, 0, 4300, 4300)),
        ("cluster_00 (35.5 x 21.2 km, big)",      (0, 0, 35500, 21200)),
        ("cluster_17 (23.9 x 36.4 km, biggest)",  (0, 0, 23900, 36400)),
        ("synthetic exact 6km",                    (0, 0, 6000, 6000)),
        ("synthetic 1.5x core",                    (0, 0, 4500, 4500)),
    ]

    for name, bbox in test_cases:
        print(f"\n==== {name} ====")
        tiles = compute_tile_grid(bbox, core_m=3000, overlap_m=3000)
        _summarize_grid(tiles, bbox)
        for t in tiles[:3]:
            print(f"  {t.tile_id}: core={t.core_bbox}, full={t.full_bbox}, edge={t.is_edge}")
        if len(tiles) > 3:
            print(f"  ... and {len(tiles)-3} more")
