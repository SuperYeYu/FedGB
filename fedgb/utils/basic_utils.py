import torch
import random
import numpy as np
import sys
from collections.abc import Iterable



def seed_everything(seed):
    """
    Sets the seed for multiple random number generators to ensure reproducibility across runs. 
    It also configures the behavior of the CUDA backend for deterministic output.

    Args:
        seed (int): The seed number to use for seeding the random number generators.

    Details:
        - Sets the seed for Python's built-in `random` module, NumPy's random module, and PyTorch.
        - Configures PyTorch's CUDA-related seeds for all GPUs.
        - Sets CUDA's cuDNN backend to operate deterministically, which can impact performance
          due to the disabling of certain optimizations like `benchmark` and general `enabled` status.

    Note:
        Enabling determinism can lead to a performance trade-off but is necessary for reproducibility
        when exact outcomes are critical to maintain across different runs, especially during debugging
        or testing phases.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    

    
    
def load_client(args, client_id, data, data_dir, message_pool, device):
    """Instantiate a public FedGB client implementation."""
    from fedgb.config.method_specs import resolve_method_class

    client_class = resolve_method_class(args.fl_algorithm, "client")
    return client_class(args, client_id, data, data_dir, message_pool, device)


def load_server(args, global_data, data_dir, message_pool, device):
    """Instantiate a public FedGB server implementation."""
    from fedgb.config.method_specs import resolve_method_class

    server_class = resolve_method_class(args.fl_algorithm, "server")
    return server_class(args, global_data, data_dir, message_pool, device)


def load_optim(args):
    """
    Loads and returns an optimizer class based on the specification in the arguments.

    Args:
        args (Namespace): Configuration arguments which include the optimizer type.

    Returns:
        An optimizer class from the `torch.optim` module.
    """
    if args.optim == "adam":
        from torch.optim import Adam
        return Adam
    
    
def load_task(args, client_id, data, data_dir, device):
    """
    Loads and returns a task instance based on the task type specified in the arguments.

    Args:
        args (Namespace): Arguments containing model and training configurations.
        client_id (int): ID of the client.
        data (object): Data specific to the client's task.
        data_dir (str): Directory containing the data.
        device (torch.device): Device to run the computations on.

    Returns:
        An instance of a task class based on the task specified.
    """
    if args.task == "node_cls":
        from fedgb.tasks.node_cls import NodeClsTask
        return NodeClsTask(args, client_id, data, data_dir, device)
    elif args.task == "graph_cls":
        from fedgb.tasks.graph_cls import GraphClsTask
        return GraphClsTask(args, client_id, data, data_dir, device)
    elif args.task == "graph_reg":
        from fedgb.tasks.graph_reg import GraphRegTask
        return GraphRegTask(args, client_id, data, data_dir, device)
    elif args.task == "link_pred":
        from fedgb.tasks.link_pred import LinkPredTask
        return LinkPredTask(args, client_id, data, data_dir, device)
    elif args.task == "node_clust":
        from fedgb.tasks.node_clust import NodeClustTask
        return NodeClustTask(args, client_id, data, data_dir, device)
    


def extract_floats(s):
    """
    Extracts and converts three floats separated by hyphens from a string and ensures their sum is 1.

    Args:
        s (str): A string containing three float numbers separated by hyphens (e.g., "0.6-0.3-0.1").

    Returns:
        tuple: A tuple of three floats (train, val, test) extracted from the string.

    Raises:
        AssertionError: If the sum of the three numbers does not equal 1.
    """
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + Decimal(parts[2]) == Decimal(1)
    return train, val, test

def idx_to_mask_tensor(idx_list, length):
    """
    Converts a list of indices to a tensor mask of a specified length.

    Args:
        idx_list (list[int]): List of indices that should be marked as 1 in the mask.
        length (int): Total length of the mask tensor.

    Returns:
        torch.Tensor: A binary mask tensor where positions corresponding to indices in idx_list are set to 1.
    """
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask



def mask_tensor_to_idx(tensor):
    """
    Converts a tensor mask to a list of indices where the tensor is non-zero.

    Args:
        tensor (torch.Tensor): A tensor containing binary values.

    Returns:
        list[int]: A list of indices corresponding to non-zero entries in the tensor.
    """
    result = tensor.nonzero().squeeze().tolist()
    if type(result) is not list:
        result = [result]
    return result
    

import sys
import torch

def total_size(o, seen=None):
    """Calculate the total memory size of a given object, avoiding infinite recursion.

    Args:
        o: The object to calculate the size of.
        seen: A set of already seen objects to avoid infinite recursion.

    Returns:
        int: The total memory size of the object in bytes.
    """
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(o, torch.Tensor):
        return o.element_size() * o.numel()
    elif isinstance(o, dict):
        return sum(total_size(v, seen) for v in o.values())
    elif isinstance(o, (list, tuple, set, frozenset)):
        return sum(total_size(i, seen) for i in o)
    elif hasattr(o, "to_dict") and callable(o.to_dict):
        try:
            return sum(total_size(v, seen) for v in o.to_dict().values())
        except Exception:
            return 0
    elif isinstance(o, (str, bytes, bytearray)):
        return sys.getsizeof(o)
    elif isinstance(o, Iterable):
        return 0
    return sys.getsizeof(o)



def model_complexity(model:torch.nn.Module):
    """
    Calculates the complexity of a PyTorch model by counting the number of parameters and computing FLOPs.

    Args:
        model (torch.nn.Module): The model for which complexity is calculated.

    Returns:
        dict: A dictionary with the total number of parameters and FLOPs.
    """
    from fvcore.nn import FlopCountAnalysis, parameter_count
    params = sum([val for val in parameter_count(model).values()])
    return params
    
