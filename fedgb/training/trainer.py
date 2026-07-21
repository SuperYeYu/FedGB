import torch
import random
from fedgb.data.distributed_dataset_loader import FGLDataset
from fedgb.utils.basic_utils import load_client, load_server
from fedgb.utils.logger import Logger
from fedgb.utils.metrics import normalize_required_metrics


class FGLTrainer:
    """
    Federated Graph Learning Trainer class to manage the training and evaluation process.

    Attributes:
        args (Namespace): Arguments containing model and training configurations.
        message_pool (dict): Dictionary to manage messages between clients and server.
        device (torch.device): Device to run the computations on.
        clients (list): List of client instances.
        server (object): Server instance.
        evaluation_result (dict): Dictionary to store the best evaluation results.
        logger (Logger): Logger instance to log training and evaluation metrics.
    """
    
    
    def __init__(self, args):
        """
        Initialize the FGLTrainer with provided arguments and dataset.

        Args:
            args (Namespace): Arguments containing model and training configurations.
        """
        self.args = args
        self.message_pool = {}
        fgl_dataset = FGLDataset(args)
        raw_num_classes = self._infer_dataset_num_classes(fgl_dataset)
        self.device = torch.device(f"cuda:{args.gpuid}" if (torch.cuda.is_available() and args.use_cuda) else "cpu")
        self.clients = [load_client(args, client_id, fgl_dataset.local_data[client_id], fgl_dataset.processed_dir, self.message_pool, self.device) for client_id in range(self.args.num_clients)]
        self.server = load_server(args, fgl_dataset.global_data, fgl_dataset.processed_dir, self.message_pool, self.device)
        self.args.metrics = normalize_required_metrics(
            self.args.metrics,
            task=self.args.task,
            num_classes=raw_num_classes or self._infer_num_classes(),
        )
        
        self.evaluation_result = {
            "best_round": 0,
            "best_val_loss": float("inf"),
            "best_test_loss": float("inf"),
        }
        if self.args.task == "graph_reg":
            for metric in self.args.metrics:
                self.evaluation_result[f"best_val_{metric}"] = float("inf")
                self.evaluation_result[f"best_test_{metric}"] = float("inf")
        elif self.args.task in ["graph_cls", "node_cls", "link_pred"]:
            for metric in self.args.metrics:
                self.evaluation_result[f"best_val_{metric}"] = 0
                self.evaluation_result[f"best_test_{metric}"] = 0
        elif self.args.task in ["node_clust"]:
            for metric in self.args.metrics:
                self.evaluation_result[f"best_{metric}"] = 0
        

        self.logger = Logger(args, self.message_pool, fgl_dataset.processed_dir, self.server.personalized)

    def _infer_dataset_num_classes(self, fgl_dataset):
        if self.args.task not in ["graph_cls", "node_cls"]:
            return None
        candidates = []
        for data in [fgl_dataset.global_data] + list(fgl_dataset.local_data):
            if data is None:
                continue
            for attr in ("num_global_classes", "num_classes"):
                try:
                    candidates.append(int(getattr(data, attr)))
                except (AttributeError, TypeError, ValueError):
                    pass
            if hasattr(data, "y") and data.y is not None:
                labels = data.y.view(-1)
                labels = labels[labels >= 0]
                if labels.numel() > 0:
                    candidates.append(int(labels.max().item()) + 1)
        return max(candidates) if candidates else None

    def _infer_num_classes(self):
        if self.args.task not in ["graph_cls", "node_cls"]:
            return None
        candidates = []
        for task in [getattr(self.server, "task", None)] + [client.task for client in self.clients]:
            if task is None:
                continue
            try:
                candidates.append(int(task.num_global_classes))
            except (AttributeError, TypeError, ValueError):
                pass
            if not hasattr(task, "data"):
                continue
            data = task.data
            for attr in ("num_global_classes", "num_classes"):
                try:
                    candidates.append(int(getattr(data, attr)))
                except (AttributeError, TypeError, ValueError):
                    pass
            if hasattr(data, "y") and data.y is not None:
                labels = data.y.view(-1)
                labels = labels[labels >= 0]
                if labels.numel() > 0:
                    candidates.append(int(labels.max().item()) + 1)
        return max(candidates) if candidates else None
        
  
    def train(self):
        """
        Train the model over a specified number of rounds, performing federated learning with the clients.
        """
        for round_id in range(self.args.num_rounds):
            sampled_clients = sorted(random.sample(list(range(self.args.num_clients)), int(self.args.num_clients * self.args.client_frac)))
            print(f"round # {round_id}\t\tsampled_clients: {sampled_clients}")
            self.message_pool["round"] = round_id
            self.message_pool["sampled_clients"] = sampled_clients
            self.server.send_message()
            for client_id in sampled_clients:
                self.clients[client_id].execute()
                self.clients[client_id].send_message()
            self.server.execute()
            if hasattr(self.server, "sync_clients_after_aggregation"):
                self.server.sync_clients_after_aggregation(self.clients)
            
            self.evaluate()
            print("-"*50)
            
        self.logger.save()
        
        
        
    def evaluate(self):
        """
        Evaluate the model based on the specified evaluation mode and task.

        Raises:
            ValueError: If the evaluation mode is not supported by the personalized algorithm.
        """
        # download -> local-train -> evaluate on local data
        evaluation_result = {"current_round": self.message_pool["round"]}
        
        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            evaluation_result["current_val_loss"] = 0
            evaluation_result["current_test_loss"] = 0
            for metric in self.args.metrics:
                evaluation_result[f"current_val_{metric}"] = 0
                evaluation_result[f"current_test_{metric}"] = 0
        elif self.args.task in ["node_clust"]:
            for metric in self.args.metrics:
                evaluation_result[f"current_{metric}"] = 0

        tot_samples = 0
        val_tot_samples = 0
        test_tot_samples = 0
        one_time_infer = False
        
        
        for client_id in range(self.args.num_clients):
            if self.args.evaluation_mode == "local_model_on_local_data":
                num_samples = self.clients[client_id].task.num_samples
                result = self.clients[client_id].task.evaluate()
            elif self.args.evaluation_mode == "local_model_on_global_data":
                num_samples = self.server.task.num_samples
                result = self.clients[client_id].task.evaluate(self.server.task.splitted_data)
            elif self.args.evaluation_mode == "global_model_on_local_data":
                num_samples = self.clients[client_id].task.num_samples
                if self.server.personalized:
                    raise ValueError(f"personalized algorithm {self.args.fl_algorithm} doesn't support global model evaluation.")
                result = self.server.task.evaluate(self.clients[client_id].task.splitted_data)
            elif self.args.evaluation_mode == "global_model_on_global_data":
                num_samples = self.server.task.num_samples
                if self.server.personalized:
                    raise ValueError(f"personalized algorithm {self.args.fl_algorithm} doesn't support global model evaluation.")
                # only one-time infer
                one_time_infer = True
                result = self.server.task.evaluate()
            
            if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
                val_loss = result.get("loss_val", result.get("mae_val"))
                test_loss = result.get("loss_test", result.get("mae_test"))
                if self.args.task == "graph_reg":
                    val_weight = int(result.get("num_observed_val", num_samples))
                    test_weight = int(result.get("num_observed_test", num_samples))
                    if val_weight and val_loss is not None and torch.isfinite(torch.as_tensor(val_loss)):
                        evaluation_result["current_val_loss"] += float(val_loss) * val_weight
                        val_tot_samples += val_weight
                    if test_weight and test_loss is not None and torch.isfinite(torch.as_tensor(test_loss)):
                        evaluation_result["current_test_loss"] += float(test_loss) * test_weight
                        test_tot_samples += test_weight
                    for metric in self.args.metrics:
                        val_metric, test_metric = result[f"{metric}_val"], result[f"{metric}_test"]
                        if val_weight and torch.isfinite(torch.as_tensor(val_metric)):
                            evaluation_result[f"current_val_{metric}"] += float(val_metric) * val_weight
                        if test_weight and torch.isfinite(torch.as_tensor(test_metric)):
                            evaluation_result[f"current_test_{metric}"] += float(test_metric) * test_weight
                else:
                    if val_loss is not None and test_loss is not None:
                        evaluation_result["current_val_loss"] += float(val_loss) * num_samples
                        evaluation_result["current_test_loss"] += float(test_loss) * num_samples
                    for metric in self.args.metrics:
                        val_metric, test_metric = result[f"{metric}_val"], result[f"{metric}_test"]
                        evaluation_result[f"current_val_{metric}"] += val_metric * num_samples
                        evaluation_result[f"current_test_{metric}"] += test_metric * num_samples
            elif self.args.task in ["node_clust"]:
                for metric in self.args.metrics:
                    metric_value = result[f"{metric}"]
                    evaluation_result[f"current_{metric}"] += metric_value * num_samples
                
            if one_time_infer:
                tot_samples = num_samples
                break
            else:
                tot_samples += num_samples
        
        
        
        if self.args.task in ["graph_cls", "graph_reg", "node_cls", "link_pred"]:
            val_denominator = val_tot_samples if self.args.task == "graph_reg" else tot_samples
            test_denominator = test_tot_samples if self.args.task == "graph_reg" else tot_samples
            if val_denominator == 0 or test_denominator == 0:
                raise ValueError("Evaluation produced no observed validation or test targets.")
            evaluation_result["current_val_loss"] /= val_denominator
            evaluation_result["current_test_loss"] /= test_denominator
            for metric in self.args.metrics:
                evaluation_result[f"current_val_{metric}"] /= val_denominator
                evaluation_result[f"current_test_{metric}"] /= test_denominator
                
            primary_metric = self.args.metrics[0]
            if self.args.task == "graph_reg":
                is_better = evaluation_result[f"current_val_{primary_metric}"] < self.evaluation_result[f"best_val_{primary_metric}"]
            else:
                is_better = evaluation_result[f"current_val_{primary_metric}"] > self.evaluation_result[f"best_val_{primary_metric}"]
            if is_better:
                self.evaluation_result["best_val_loss"] = evaluation_result["current_val_loss"]
                self.evaluation_result["best_test_loss"] = evaluation_result["current_test_loss"]
                for metric in self.args.metrics:
                    self.evaluation_result[f"best_val_{metric}"] = evaluation_result[f"current_val_{metric}"]
                    self.evaluation_result[f"best_test_{metric}"] = evaluation_result[f"current_test_{metric}"]
                self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]
            
            current_output = f"curr_round: {evaluation_result['current_round']}\t" + \
                "\t".join([f"curr_val_{metric}: {evaluation_result[f'current_val_{metric}']:.4f}\tcurr_test_{metric}: {evaluation_result[f'current_test_{metric}']:.4f}" for metric in self.args.metrics]) + \
                f"\tcurr_val_loss: {evaluation_result['current_val_loss']:.4f}\tcurr_test_loss: {evaluation_result['current_test_loss']:.4f}"
        
            best_output = f"best_round: {self.evaluation_result['best_round']}\t" + \
                "\t".join([f"best_val_{metric}: {self.evaluation_result[f'best_val_{metric}']:.4f}\tbest_test_{metric}: {self.evaluation_result[f'best_test_{metric}']:.4f}" for metric in self.args.metrics]) + \
                f"\tbest_val_loss: {self.evaluation_result['best_val_loss']:.4f}\tbest_test_loss: {self.evaluation_result['best_test_loss']:.4f}"
    
            print(current_output)
            print(best_output)
        else:
            for metric in self.args.metrics:
                evaluation_result[f"current_{metric}"] /= tot_samples
        
            if evaluation_result[f"current_{self.args.metrics[0]}"] > self.evaluation_result[f"best_{self.args.metrics[0]}"]:
                for metric in self.args.metrics:
                    self.evaluation_result[f"best_{metric}"] = evaluation_result[f"current_{metric}"]
                self.evaluation_result[f"best_round"] = evaluation_result[f"current_round"]
            
            current_output = f"curr_round: {evaluation_result['current_round']}\t" + \
                "\t".join([f"curr_{metric}: {evaluation_result[f'current_{metric}']:.4f}" for metric in self.args.metrics])
        
            best_output = f"best_round: {self.evaluation_result['best_round']}\t" + \
                "\t".join([f"best_{metric}: {self.evaluation_result[f'best_{metric}']:.4f}" for metric in self.args.metrics])
        
            print(current_output)
            print(best_output)
            
        self.logger.add_log(evaluation_result)
            
    
        
        
        
     
