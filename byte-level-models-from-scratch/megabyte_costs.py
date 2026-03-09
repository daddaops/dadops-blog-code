"""MegaByte Cost Analysis — attention cost savings from patching.

Code Block 4: Shows how MegaByte's hierarchical approach reduces compute.
"""

def megabyte_costs(N, P):
    """Compute attention operations for MegaByte vs flat transformer."""
    flat_attn = N * N           # O(N^2) for flat byte transformer
    global_attn = (N // P) ** 2 # O((N/P)^2) for global model
    local_attn = N * P          # O(N*P) total for all local models
    mega_attn = global_attn + local_attn
    return flat_attn, mega_attn

print(f"{'Seq Len':>8} {'Flat':>12} {'MegaByte':>12} {'Speedup':>8}")
print("-" * 44)
for N in [512, 1024, 2048, 4096, 8192]:
    flat, mega = megabyte_costs(N, P=8)
    print(f"{N:>8} {flat:>12,} {mega:>12,} {flat/mega:>7.1f}x")

# Blog claims:
#     512      262,144        8,192    32.0x
#    1024    1,048,576       24,576    42.7x
#    2048    4,194,304       81,920    51.2x
#    4096   16,777,216      294,912    56.9x
#    8192   67,108,864    1,114,112    60.2x

# MegaByte's advantage grows with sequence length!
# Patch embedding: concatenate P bytes, project through linear layer
P = 8
d_model = 512
patch_embed_params = P * 256 + P * d_model  # byte embeds + projection
print(f"\nPatch embedding: {patch_embed_params:,} params")
print(f"Equivalent to a {P}-gram byte encoder")
