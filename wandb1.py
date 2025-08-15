# test_wandb.py
import wandb

wandb.login()
run = wandb.init(project="test-project", name="test-run")
run.log({"acc": 0.9})
run.finish()