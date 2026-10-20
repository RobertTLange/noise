"""Pallas GPU kernels of the trainer (forward-only: nothing here needs a gradient rule).

Each kernel has a plain-jnp reference next to it, the one it is checked against. The run log embeds the source of every
kernel module, so a record stays reproducible from its log.
"""
