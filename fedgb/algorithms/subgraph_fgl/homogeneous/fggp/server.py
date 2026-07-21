import torch
from fedgb.training.base import BaseServer
from fedgb.algorithms.subgraph_fgl.homogeneous.fggp.fggp_config import config
from fedgb.algorithms.subgraph_fgl.homogeneous.fggp.models import FedGCN
from fedgb.algorithms.subgraph_fgl.homogeneous.fggp.utils import aggregate_fggp_prototypes


class FGGPServer(BaseServer):
    """
    FGGPServer is the server-side implementation for the Federated Graph Learning with Generalizable Prototypes 
    (FGGP) framework. The server aggregates model parameters and prototypes from clients and performs 
    prototype clustering using the FINCH algorithm to generate generalizable prototypes across federated clients.

    Attributes:
        global_protos (dict): A dictionary containing the aggregated global prototypes for each class.
    """
    
    
    
    def __init__(self, args, global_data, data_dir, message_pool, device):
        """
        Initializes the FGGPServer.

        Args:
            args (Namespace): Arguments containing model and training configurations.
            global_data (torch_geometric.data.Data): Global graph data available to the server, if any.
            data_dir (str): Directory containing the data.
            message_pool (dict): Pool for managing messages between client and server.
            device (torch.device): The device on which computations will be performed (e.g., CPU or GPU).
        """
        super(FGGPServer, self).__init__(args, global_data, data_dir, message_pool, device)
        self.task.load_custom_model(FedGCN(nfeat=self.task.num_feats, nhid=self.args.hid_dim,
                                           nclass=self.task.num_global_classes, nlayer=self.args.num_layers,
                                           dropout=self.args.dropout))
        self.global_protos = {}



    def execute(self):
        """
        Executes the global aggregation of model parameters and prototype aggregation from all sampled clients.
        The model parameters are aggregated based on either the number of samples or equally, depending on the 
        configuration. The prototypes from clients are aggregated using the FINCH algorithm.
        """
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in
                                   self.message_pool[f"sampled_clients"]])
            for it, client_id in enumerate(self.message_pool["sampled_clients"]):
                if config["params_weight"] == "samples_num":
                    weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                else:
                    weight = 1/len(self.message_pool["sampled_clients"])

                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"],
                                                       self.task.model.parameters()):
                    if it == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
        self.global_protos = self.proto_aggregation()



    def send_message(self):
        """
        Sends the aggregated global model parameters to the clients.
        """
        self.message_pool["server"] = {
            "weight": list(self.task.model.parameters()),
            "global_protos": self.global_protos,
        }
        
        

    def proto_aggregation(self):
        """
        Aggregates the prototypes received from clients. For each class, the prototypes are clustered using the 
        FINCH algorithm to find representative prototypes. The resulting prototypes are averaged for each cluster.

        Returns:
            dict: A dictionary containing the aggregated prototypes for each class.
        """
        local_protos_list = [
            self.message_pool[f"client_{idx}"]["protos"]
            for idx in self.message_pool["sampled_clients"]
        ]
        return aggregate_fggp_prototypes(
            local_protos_list,
            self.device,
            use_finch=config["fggp_use_finch"],
        )
