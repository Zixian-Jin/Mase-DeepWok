
import numpy as np
from functools import lru_cache 

from chop.ir.graph import MaseMetadata

from .common import Shard
from .mesh import Mesh

BYTES_PER_ELEMENT = 4

def get_communication_cost(sharding: tuple, node_meta: MaseMetadata, mesh: Mesh):
    assert sharding[0][1] == sharding[1][0], f"Inconsistent sharding for node: {node_meta.node}"
    inner_dim_sharding = sharding[1][0]

    out_shape = node_meta["common"]["results"]["data_out_0"]["shape"]

    if inner_dim_sharding == Shard.R:
        return 0
    
    else:
        ar_dim = inner_dim_sharding.value # 0 for S_0, 1 for S_1
        return mesh.all_reduce_cost(num_bytes = BYTES_PER_ELEMENT * np.prod(out_shape), mesh_dim = ar_dim)

@lru_cache(maxsize=None)
def get_resharding_cost(mesh: Mesh, src: tuple, dest: tuple, dest_node_meta: MaseMetadata):
    """
    Obtain the resharding cost given a source and destination sharding profile for a tensor.
    The mesh object is assumed to have been initialized with alpha, beta parameters so that
    the communication cost can be estimated for each MPI operator.
    """


    # If original sharding is fully replicated, no resharding is required
    if src == dest or src == (Shard.R, Shard.R):
        return 0
   
    num_bytes = BYTES_PER_ELEMENT * np.prod(dest_node_meta["common"]["args"]["data_in_0"]["shape"])
    
    # No cost (simple split along given mesh dimension)
    if (
            # Keep dim 0, split dim 1
            # E.g. (R, R) -> (R, S_0), (S_0, R) -> (S_0, S_1)
            (src[0] == dest[0]) and (src[1] == Shard.R) and (dest[1] in [Shard.S_0, Shard.S_1])
            # Split dim 0, keep dim 1
            # E.g. (R, R) -> (S_1, R), (R, S_1) -> (S_0, S_1)
            or (src[1] == dest[1]) and (src[0] == Shard.R) and (dest[0] in [Shard.S_0, Shard.S_1])
        ):
        return 0

    # Split -> Replicate (All Gather)
    elif (
            # Keep dim 0, gather along dim 1
            # E.g. (S_1, S_0) -> (S_1, R)
            (src[0] == dest[0]) and (src[1] in [Shard.S_0, Shard.S_1]) and (dest[1] == Shard.R)
            # Gather along dim 0, keep dim 1
            # E.g. (S_0, S_1) -> (R, S_1)
            or (src[1] == dest[1]) and (src[0] in [Shard.S_0, Shard.S_1]) and (dest[0] == Shard.R)
        ):
        ag_dim = 1 if src[0] == dest[0] else 0
        return mesh.all_gather_cost(
            num_bytes = num_bytes,
            mesh_dim = ag_dim,
        )

    # All-to-all
    # E.g. (R, S_0) -> (S_0, R), (S_1, R) -> (R, S_1)
    elif (src[0] == dest[1] and src[1] == dest[0] and (Shard.R in src)):
        # all to all
        a2a_dim = src[0].value if src[0] != Shard.R else src[1].value
        return mesh.all_to_all_cost(
            num_bytes = num_bytes,
            mesh_dim = a2a_dim,
        )

    # Two-stage resharding: when the resharding cannot be resolved with a single split, all-gather or all-to-all,
    # must first gather along the first non-replicated dimension, then recursively compute the cost for the
    # reduced sharding
    else:
        # Reduce one dimension and re-compute
        if (src[0] != Shard.R):
            new_src = (Shard.R, src[1])
            ag_dim = src[0].value
        else:
            new_src = (Shard.R, Shard.R)
            ag_dim = src[1].value

        return mesh.all_gather_cost(
            num_bytes = num_bytes, 
            mesh_dim = ag_dim
        ) + get_resharding_cost(mesh, new_src, dest, dest_node_meta)

def get_resharding_matrix(mesh, src_shardings, dest_shardings, dest_node_meta):
    mat = np.zeros((len(dest_shardings), len(src_shardings)))
    for src_idx, src in enumerate(src_shardings):
        for dest_idx, dest in enumerate(dest_shardings):
            mat[dest_idx, src_idx] = get_resharding_cost(mesh, src, dest, dest_node_meta)

    return mat
