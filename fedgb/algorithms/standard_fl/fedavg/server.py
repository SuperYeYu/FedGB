import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.standard_fl.fedavg.utils import shared_parameter_payload

class FedAvgServer(BaseServer):
    """
    FedAvgServer implements the server-side logic for the Federated Averaging (FedAvg) algorithm,
    as introduced in the paper "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al. (2017). This class is responsible for aggregating model updates from clients
    and broadcasting the updated global model to all participants in the federated learning process.

    Attributes:
        None (inherits attributes from BaseServer)
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedAvgServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedAvgServer, self).__init__(args, global_data, data_dir, message_pool, device)

   
    def execute(self):
        """
        Executes the server-side operations. This method aggregates model updates from the 
        clients by computing a weighted average of the model parameters, based on the number
        of samples each client used for training.
        """
        with torch.no_grad():
            sampled_clients = [
                client_id
                for client_id in self.message_pool["sampled_clients"]
                if self.message_pool[f"client_{client_id}"]["num_samples"] > 0
            ]
            if not sampled_clients:
                return
            num_tot_samples = sum(
                self.message_pool[f"client_{client_id}"]["num_samples"]
                for client_id in sampled_clients
            )
            named_params = dict(self.task.model.named_parameters())
            for it, client_id in enumerate(sampled_clients):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                client_message = self.message_pool[f"client_{client_id}"]
                weight_names = client_message.get("weight_names")
                if weight_names is None:
                    weight_names = [name for name, _ in self.task.model.named_parameters()]

                for name, local_param in zip(weight_names, client_message["weight"]):
                    global_param = named_params[name]
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        
    def send_message(self):
        """
        Sends a message to the clients containing the updated global model parameters after 
        aggregation.
        """
        private_head = getattr(self.args, "private_head", False) or getattr(self.args, "fedavg_private_head", False)
        weight_names, weights = shared_parameter_payload(self.task.model, getattr(self.args, "task", None), private_head)
        self.message_pool["server"] = {
            "weight_names": weight_names,
            "weight": weights
        }
