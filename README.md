# large-scale-dpsgd
Investigating large scale dpsgd

parser:

--model_name', type=str, default="scatternet_cnn", choices=["alexnet", "scatternet_cnn", "resnet18"]

'--batch_size', type=int, default=256

'--val_batch_size', type=int, default=64

'--mini_batch_size', type=int, default=2

'--lr', type=float, default=0.01

'--optim', type=str, default="SGD", choices=["SGD", "Adam"]

'--momentum', type=float, default=0.9

'--noise_multiplier', type=float, default=1.3

'--max_grad_norm', type=float, default=1

'--epochs', type=int, default=10
    
