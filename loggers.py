from abc import ABC, abstractmethod
import os
from datetime import datetime


class AbstractLogger(ABC):
    """Abstract base class for experiment logging."""
    
    @abstractmethod
    def log_scalar(self, tag, value, step):
        """Log a scalar value."""
        pass
    
    @abstractmethod
    def log_histogram(self, tag, values, step):
        """Log histogram data."""
        pass
    
    @abstractmethod
    def log_image(self, tag, image, step):
        """Log an image."""
        pass
    
    @abstractmethod
    def log_text(self, tag, text, step):
        """Log text."""
        pass
    
    @abstractmethod
    def log_hyperparams(self, params):
        """Log hyperparameters."""
        pass
    
    @abstractmethod
    def flush(self):
        """Flush logs to disk/server."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the logger."""
        pass


class TensorboardLogger(AbstractLogger):
    """TensorBoard implementation of the AbstractLogger."""
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("TensorBoard not found. Install with 'pip install tensorboard'")
        
        if log_dir is None:
            log_dir = os.path.join("runs", "tensorboard")
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logs will be saved to {self.log_dir}")
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram data to TensorBoard."""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, image, step):
        """Log an image to TensorBoard."""
        self.writer.add_image(tag, image, step)
    
    def log_text(self, tag, text, step):
        """Log text to TensorBoard."""
        self.writer.add_text(tag, text, step)
    
    def log_hyperparams(self, params):
        """Log hyperparameters to TensorBoard."""
        from torch.utils.tensorboard.summary import hparams
        self.writer.add_hparams(params, {})
    
    def flush(self):
        """Flush logs to disk."""
        self.writer.flush()
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class WandbLogger(AbstractLogger):
    """Weights & Biases implementation of the AbstractLogger."""
    
    def __init__(self, project=None, entity=None, name=None, config=None):
        """
        Initialize Wandb logger.
        
        Args:
            project (str): W&B project name
            entity (str): W&B team/entity name
            name (str): Run name
            config (dict): Configuration parameters for the run
        """
        try:
            import wandb
        except ImportError:
            raise ImportError("Wandb not found. Install with 'pip install wandb'")
        
        if project is None:
            project = "dqn-tdlambda"
            
        if name is None:
            name = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.wandb = wandb
        self.run = wandb.init(project=project, entity=entity, name=name, config=config)
        print(f"Wandb run '{name}' initialized in project '{project}'")
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to Wandb."""
        self.wandb.log({tag: value}, step=step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram data to Wandb."""
        self.wandb.log({tag: self.wandb.Histogram(values)}, step=step)
    
    def log_image(self, tag, image, step):
        """Log an image to Wandb."""
        self.wandb.log({tag: self.wandb.Image(image)}, step=step)
    
    def log_text(self, tag, text, step):
        """Log text to Wandb."""
        self.wandb.log({tag: self.wandb.Html(text)}, step=step)
    
    def log_hyperparams(self, params):
        """Log hyperparameters to Wandb."""
        self.run.config.update(params)
    
    def flush(self):
        """Flush logs to Wandb server."""
        # Wandb automatically syncs data
        pass
    
    def close(self):
        """Finish the Wandb run."""
        self.wandb.finish()


class LoggerFactory:
    """Factory class to create the appropriate logger."""
    
    @staticmethod
    def create_logger(logger_type, **kwargs):
        """
        Create a logger of the specified type.
        
        Args:
            logger_type (str): Type of logger ('tensorboard' or 'wandb')
            **kwargs: Arguments to pass to the logger constructor
            
        Returns:
            AbstractLogger: An instance of the requested logger
        """
        if logger_type.lower() == 'tensorboard':
            return TensorboardLogger(**kwargs)
        elif logger_type.lower() in ['wandb', 'weights_and_biases']:
            return WandbLogger(**kwargs)
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")