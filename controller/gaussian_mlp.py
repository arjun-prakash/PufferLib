import numpy as np
from typing import Any, Optional, List
from collections import defaultdict

from code_blocks import (
    headers_network_evaluate,
    linear_activation,
    sigmoid_activation,
    relu_activation,
)

def generate(policy: Any, output_path: Optional[str] = None, desired_output_dim: Optional[int] = 4) -> str:
    """
    Generate C for an MLP policy from a PyTorch nn.Module or raw state_dict.
    - Hidden layers: tanh
    - Output layer: linear
    - Handles multi-head (e.g., value head) by choosing the chain whose final out_dim
      matches desired_output_dim (default 4). If not found, chooses the longest chain,
      then the one with largest final out_dim.

    Args:
        policy: torch.nn.Module or state_dict (dict)
        output_path: optional file path to write the C code
        desired_output_dim: prefer a chain that ends with this out_dim (default 4)
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required")

    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t, dtype=np.float32)

    # ---- Extract candidate linear layers (order-agnostic) ----
    layers_raw = []  # list of dicts: {name, W[in,out], b[out], in_dim, out_dim}

    if isinstance(policy, nn.Module):
        for name, sub in policy.named_modules():
            if isinstance(sub, nn.Linear):
                W = to_numpy(sub.weight).T.astype(np.float32)   # [in, out]
                b = to_numpy(sub.bias).astype(np.float32) if sub.bias is not None else np.zeros(W.shape[1], np.float32)
                layers_raw.append(dict(name=name, W=W, b=b, in_dim=W.shape[0], out_dim=W.shape[1]))
    elif isinstance(policy, dict):
        sd = policy
        # collect all *.weight with 2D tensors
        prefixes = {k.rsplit('.', 1)[0] for k in sd.keys() if k.endswith('.weight')}
        for p in prefixes:
            wk, bk = f"{p}.weight", f"{p}.bias"
            if wk in sd:
                Wt = to_numpy(sd[wk])
                if Wt.ndim != 2:
                    continue
                W = Wt.T.astype(np.float32)                    # [in, out]
                b = to_numpy(sd[bk]).astype(np.float32) if bk in sd else np.zeros(W.shape[1], np.float32)
                layers_raw.append(dict(name=p, W=W, b=b, in_dim=W.shape[0], out_dim=W.shape[1]))
    else:
        raise TypeError("policy must be a torch.nn.Module or a state_dict (dict)")

    if not layers_raw:
        raise ValueError("No Linear-like layers found.")

    # ---- Build all chains by matching out_dim → in_dim (topological order) ----
    by_in = defaultdict(list)
    by_out = defaultdict(list)
    for i, L in enumerate(layers_raw):
        by_in[L["in_dim"]].append(i)
        by_out[L["out_dim"]].append(i)

    # starts: layers whose in_dim is not any other layer's out_dim
    starts = [i for i, L in enumerate(layers_raw) if L["in_dim"] not in by_out]

    # If we didn't find a start (rare, but possible with weird saves), just fall back to heuristic:
    if not starts:
        # Try to find the smallest in_dim as a likely input layer
        min_in_idx = min(range(len(layers_raw)), key=lambda k: layers_raw[k]["in_dim"])
        starts = [min_in_idx]

    # DFS to enumerate chains (handles branching heads)
    chains = []

    def extend(path):
        cur = path[-1]
        cur_out = layers_raw[cur]["out_dim"]
        next_idxs = [j for j in by_in.get(cur_out, []) if j not in path]
        if not next_idxs:
            chains.append(path)
            return
        for nx in next_idxs:
            extend(path + [nx])

    for s in starts:
        extend([s])

    # Deduplicate identical chains
    uniq = []
    seen = set()
    for ch in chains:
        t = tuple(ch)
        if t not in seen:
            seen.add(t)
            uniq.append(chains[chains.index(ch)])
    chains = uniq

    # ---- Pick best chain ----
    def chain_score(path):
        # prefer desired_output_dim at end, then longer chain, then larger final out_dim
        end_out = layers_raw[path[-1]]["out_dim"]
        has_desired = (desired_output_dim is not None and end_out == desired_output_dim)
        return (
            1 if has_desired else 0,
            len(path),
            end_out
        )

    if not chains:
        # Fall back to picking a single layer with smallest in_dim then choose successor by size
        chains = [[min(range(len(layers_raw)), key=lambda k: layers_raw[k]["in_dim"])]]

    best = max(chains, key=chain_score)
    chain_layers = [layers_raw[i] for i in best]

    # ---- Now we have a proper ordered chain: input → ... → output ----
    L = len(chain_layers)
    structure = f"static const int structure[{L}][2] = {{"
    weights_c, biases_c, outputs_c = [], [], []

    for idx, Lr in enumerate(chain_layers):
        in_dim, out_dim = Lr["in_dim"], Lr["out_dim"]
        structure += f"{{{in_dim}, {out_dim}}},"
        outputs_c.append(f"static float output_{idx}[{out_dim}];\n")

        # weights to C
        rows = [ "{" + ",".join(f"{float(Lr['W'][r, c]):.8g}" for c in range(out_dim)) + "}" for r in range(in_dim) ]
        weights_c.append(
            f"static const float layer_{idx}_weight[{in_dim}][{out_dim}] = "
            f"{{{','.join(rows)}}};\n"
        )

        # biases to C
        biases_c.append(
            f"static const float layer_{idx}_bias[{out_dim}] = "
            "{" + ",".join(f"{float(v):.8g}" for v in Lr["b"]) + "};\n"
        )

    structure = structure[:-1] + "};\n"

    # ---- Forward loops ----
    for_loops = []
    for idx in range(L):
        src = "state_array" if idx == 0 else f"output_{idx-1}"
        loop = f"""
        for (int i = 0; i < structure[{idx}][1]; i++) {{
            output_{idx}[i] = 0;
            for (int j = 0; j < structure[{idx}][0]; j++) {{
                output_{idx}[i] += {src}[j] * layer_{idx}_weight[j][i];
            }}
            output_{idx}[i] += layer_{idx}_bias[i];"""
        if idx < L - 1:  # hidden layers tanh
            loop += f"""
            output_{idx}[i] = tanhf(output_{idx}[i]);"""
        loop += """
        }
"""
        for_loops.append(loop)

    # ---- Assign to control_n (up to 4 outputs) ----
    last_out_dim = chain_layers[-1]["out_dim"]
    assign_count = min(4, last_out_dim)
    if assign_count == 0:
        raise ValueError("Final chain has zero outputs.")
    assignment = "\n".join(
        [f"\t\tcontrol_n->thrust_{k} = output_{L-1}[{k}];" for k in range(assign_count)]
    ) + "\n"

    controller_eval = "	void networkEvaluate(struct control_t_n *control_n, const float *state_array){\n"
    controller_eval += "".join(for_loops)
    controller_eval += assignment
    controller_eval += "}\n"

    source = (
        headers_network_evaluate +
        linear_activation +
        sigmoid_activation +
        relu_activation +
        structure +
        "".join(outputs_c) +
        "".join(weights_c) +
        "".join(biases_c) +
        controller_eval
    )

    if output_path:
        with open(output_path, "w") as f:
            f.write(source)

    return source
