import torch
from fedgb.training.base import BaseServer

class FedProtoServer(BaseServer):
    """
    FedProtoServer is a server implementation for the Federated Prototype Learning (FedProto) framework. 
    This server is responsible for aggregating local prototypes from clients to update the global prototypes, 
    which are then used in the federated learning process.

    Attributes:
        global_prototype (dict): A dictionary storing the global prototypes for each class, updated 
                                 based on the local prototypes received from the clients.
    """
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FedProtoServer.

        Attributes:
            args (Namespace): Arguments containing model and training configurations.
            global_data (object): Global dataset accessible by the server.
            data_dir (str): Directory containing the data.
            message_pool (object): Pool for managing messages between server and clients.
            device (torch.device): Device to run the computations on.
        """
        super(FedProtoServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.global_prototype = {}
   
   
   
    def _prototype_ids(self):
        if self.args.task == "graph_reg":
            return [0]
        return range(self.task.num_global_classes)

    def execute(self):
        """
        Executes the server-side operations for aggregating local prototypes from clients. 
        The global prototypes for each class are computed as the weighted average of the 
        local prototypes from the sampled clients.
        """
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in self.message_pool[f"sampled_clients"]])
            weighting = str(getattr(self.args, "fedproto_proto_weighting", "sample")).lower()
            for class_i in self._prototype_ids():
                class_weights = []
                if weighting == "class_count":
                    total_class_samples = 0
                    for client_id in self.message_pool["sampled_clients"]:
                        class_count = self.message_pool[f"client_{client_id}"].get("prototype_counts", {}).get(class_i, 0)
                        total_class_samples += class_count
                    for client_id in self.message_pool["sampled_clients"]:
                        class_count = self.message_pool[f"client_{client_id}"].get("prototype_counts", {}).get(class_i, 0)
                        class_weights.append(0.0 if total_class_samples == 0 else class_count / total_class_samples)
                elif weighting == "uniform":
                    class_weights = [1.0 / len(self.message_pool["sampled_clients"])] * len(self.message_pool["sampled_clients"])
                else:
                    class_weights = [self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples for client_id in self.message_pool["sampled_clients"]]

                for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                    weight = class_weights[it]
                    proto = self.message_pool[f"client_{client_id}"]["local_prototype"][class_i]
                    if it == 0:
                        self.global_prototype[class_i] = weight * proto
                    else:
                        self.global_prototype[class_i] += weight * proto
            
        
        
    def send_message(self):
        """
        Sends a message to the clients containing the updated global prototypes. These prototypes 
        are used by the clients in their local training processes to ensure alignment with the global model.
        """
        self.message_pool["server"] = {
            "global_prototype": self.global_prototype
        }