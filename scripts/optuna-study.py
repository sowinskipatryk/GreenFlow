import os
import sys
import optuna
import logging
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sumo_rl import SumoEnvironment
from optuna.integration import OptunaPruning

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("optuna-study.log"),
                        logging.StreamHandler()])

def environment_setup():
    logging.info("Setting up SUMO environment.")
    route_files = (
            "../simulation/demand/car_ev.rou.xml,"
            "../simulation/demand/car.rou.xml,"
            "../simulation/demand/emergency.rou.xml,"
            "../simulation/demand/motorcycle.rou.xml,"
            "../simulation/demand/bus.rou.xml,"
            "../simulation/demand/tram.rou.xml,"
            "../simulation/demand/truck.rou.xml")

    env = SumoEnvironment(
        net_file='../simulation/network/osm.net.xml',  # Net file
        route_file=route_files,
        out_csv_name='../models/optuna_ppo',
        additional_sumo_cmd="--collision.action remove --ignore-route-errors ",
        single_agent=True,
        ts_ids =['Glowny_wezel'],
        use_gui=False,
        num_seconds=3600,
        reward_fn='diff-waiting-time')
    logging.info("SUMO environment created.")
    return env

def objective(trial):
    """
    Objective function for Optuna study.
    """
    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [2048, 4096, 8192])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.2)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    net_arch = trial.suggest_categorical('net_arch', ["tiny", "small", "medium"])
    net_arch_dict = {
        "tiny": [64, 64],
        "small": [256, 128],
        "medium": [256, 128, 64]
    }
    
    # Setup environment
    env = environment_setup()

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        batch_size=batch_size,
        policy_kwargs=dict(net_arch=net_arch_dict[net_arch]),
        tensorboard_log="../models/optuna_tensorboard/"
    )

    # Train and evaluate model
    eval_env = environment_setup()
    pruning_callback = OptunaPruning(trial, "mean_reward")
    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=pruning_callback,
        best_model_save_path=f'../models/optuna/trial_{trial.number}/',
        log_path=f'../models/optuna_tensorboard/trial_{trial.number}/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    try:
        model.learn(total_timesteps=30000, callback=eval_callback)
    except (AssertionError, ValueError) as e:
        logging.error(e)
        raise optuna.exceptions.TrialPruned()
    finally:
        env.close()
        eval_env.close()

    last_mean_reward = eval_callback.last_mean_reward
    
    return last_mean_reward

def main():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10000)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    try:
        study.optimize(objective, n_trials=50, timeout=7200)
    except KeyboardInterrupt:
        pass


    logging.info("Number of finished trials: ", len(study.trials))
    logging.info("Best trial:")
    trial = study.best_trial

    logging.info("  Value: ", trial.value)
    logging.info("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open("best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    logging.info("Best parameters saved to best_params.json")

if __name__ == "__main__":
    main()
