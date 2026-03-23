import os
import sys
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sumo_rl import SumoEnvironment, env

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("rl-simulation.log"),
                        logging.StreamHandler()
                    ])

#sumo-rl documentation: https://lucasalegre.github.io/sumo-rl/

def check_sumo_home():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
        logging.info("SUMO_HOME found and tools added to path.")
    else:
        logging.error("SUMO_HOME environment variable not set. Please set it to your SUMO installation directory.")
        sys.exit("Error: SUMO_HOME environment variable not set. Please set it to your SUMO installation directory.")


PT_VEHICLE_TYPES = {'bus', 'tram_gdansk'}
PT_WAIT_CAP = 60.0        # maximum PT waiting time considered in penalty (seconds)
PT_WAIT_MULTIPLIER = 2.0  # multiplier for PT waiting time penalty (added not to spoil the function composition that sums up to 1)
PT_WAIT_NORM = 100.0      # normalization factor for total PT waiting time
WAIT_NORM = 100.0         # expected max waiting time change per step (seconds)


def baltycka_reward_fn(ts) -> float:
    # 1. Waiting time difference between current and previous step
    current_wait = sum(ts.get_accumulated_waiting_time_per_lane())
    last_wait = getattr(ts, '_last_wait', current_wait)
    waiting_time_delta = np.clip((last_wait - current_wait) / WAIT_NORM, -1, 1)  # clipped so it won't dominate the rest
    ts._last_wait = current_wait

    # 2. Quadratic queue penalty (normalized by number of lanes)
    queues = ts.get_lanes_queue()
    queue_penalty = -sum(q ** 2 for q in queues) / len(queues)

    # 3. Average vehicle speed (already normalized by sumo-rl)
    avg_speed = ts.get_average_speed()

    # 4. Public transport priority (penalty for buses/trams waiting)
    pt_penalty = 0.0
    for veh_id in ts._get_veh_list():
        if ts.sumo.vehicle.getTypeID(veh_id) in PT_VEHICLE_TYPES:
            wait = ts.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
            pt_penalty -= min(wait, PT_WAIT_CAP) * PT_WAIT_MULTIPLIER
    pt_penalty /= PT_WAIT_NORM

    # 5. Phase switching penalty (punishes hard rapid phase changes, gentle to rare switches)
    switch_penalty = 0.0
    now = ts.env.sim_step
    if getattr(ts, '_last_phase', None) != ts.green_phase:
        dt = now - getattr(ts, '_last_switch_time', now)
        switch_penalty = -1.0 / (dt + 1)
        ts._last_switch_time = now
    ts._last_phase = ts.green_phase

    return (
        0.30 * waiting_time_delta +
        0.25 * queue_penalty +
        0.20 * avg_speed +
        0.15 * pt_penalty +
        0.10 * switch_penalty
    )


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
        net_file='../simulation/network/osm-rl-agent.net.xml',  # Net file
        route_file=route_files,
        out_csv_name='../models/ppo_single_agent',
        additional_sumo_cmd="--collision.action remove --ignore-route-errors ",
        single_agent=True,
        ts_ids =['Glowny_wezel'],
        use_gui=False,
        num_seconds=3600,
        reward_fn=baltycka_reward_fn)
    logging.info("SUMO environment created.")
    return env

def create_model(env):
    logging.info("Initializing PPO model...")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001,
        ent_coef=0.05,
        gamma=0.99,
        n_steps=2048,
        batch_size=128,
        device='auto',
        policy_kwargs=dict(net_arch=[256, 128, 64]),
        tensorboard_log="../models/ppo_traffic_tensorboard/")
    
    logging.info("PPO model initialized.")
    return model

def model_learn(model, callback=None):
    logging.info("Starting model training.")
    model.learn(total_timesteps=100_000, callback=callback)
    logging.info("Training finished!")


def model_save(model):
    model.save("../models/ppo_galeria_baltycka_v1")
    logging.info("Model saved to ../models/ppo_galeria_baltycka_v1")

def close_environment(env):
    env.close()
    logging.info("Environment closed.")

def evaluate_model(eval_env):
    logging.info("Setting up model evaluation.")
    eval_callback = EvalCallback(
            eval_env, 
            best_model_save_path='../models/best_model/',
            log_path='../models/ppo_traffic_tensorboard/eval_logs/', 
            eval_freq=5_000, 
            deterministic=True, 
            render=False)
    logging.info("Evaluation callback created.")
    return eval_callback, eval_env

def main():
    logging.info("--- Starting RL Simulation ---")
    check_sumo_home()
    training_env = environment_setup()
    eval_env = environment_setup()
    model = create_model(training_env)

    eval_callback, eval_env = evaluate_model(eval_env)
    model_learn(model, callback=eval_callback)
    
    model_save(model)
    close_environment(training_env)
    close_environment(eval_env)
    logging.info("--- RL Simulation Finished ---")

if __name__ == "__main__":
    main()