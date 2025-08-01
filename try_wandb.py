import wandb

wandb.init(project="test-connection", name="test-run")
wandb.log({"ping": 1})
wandb.finish()
