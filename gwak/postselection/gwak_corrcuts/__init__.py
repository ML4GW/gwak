"""gwak_corrcuts — fast TF-tile coherence/correlation proxies for the GWAK postselection veto.

These are PROXIES, not exact cWB/oLIB statistics. Every cWB/oLIB-inspired feature carries a
``_tile`` (faithful, Regime A) or ``_proxy`` (degraded, Regime B) suffix; the bare cWB names
(netcc, ecor, rho, ...) are never used. Each output row carries ``regime``.

Phase 2 = vertical slice. See DESIGN.md / design_parts/implementation.md.
"""

__all__ = ["config", "io", "tiles", "matching", "features", "pipeline"]
