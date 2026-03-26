import os
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment

def run_evaluation():
    print("IInitializing evaluation environment...")

    route_files = (
        "../simulation/demand/netedit_cfg.rou.xml,"
        "../simulation/demand/car_ev.rou.xml,"
        "../simulation/demand/car.rou.xml,"
        "../simulation/demand/emergency.rou.xml,"
        "../simulation/demand/motorcycle.rou.xml,"
        "../simulation/demand/bus.rou.xml,"
        "../simulation/demand/tram.rou.xml,"
        "../simulation/demand/truck.rou.xml"
    )

    sumo_cmd = (
        "--collision.action remove "
        "--ignore-route-errors "
        "--additional-files ../simulation/osm.add.xml "
        "--emission-output ../simulation/results/evaluate_results/emissions.xml "
        "--tripinfo-output ../simulation/results/evaluate_results/tripinfos.xml "
        "--tripinfo-output.write-unfinished true "
        "--stop-output ../simulation/results/evaluate_results/stopinfos.xml "
        "--statistic-output ../simulation/results/evaluate_results/stats.xml "
        "--duration-log.statistics true"
    )

    env = SumoEnvironment(
        net_file='../simulation/network/osm-rl-agent.net.xml',  
        route_file=route_files,
        out_csv_name='../models/evaluate_results/ppo_eval',
        additional_sumo_cmd=sumo_cmd,
        single_agent=True,
        ts_ids=['Glowny_wezel'],
        use_gui=True,
        num_seconds=3600,
        reward_fn='diff-waiting-time'
    )

    model_path = "../models/best_model/best_model.zip" 
    
    if not os.path.exists(model_path):
        print(f"Error: Missing path {model_path}")
        return

    print(f"loading model from {model_path}...")
    model = PPO.load(model_path, env=env)

    obs, info = env.reset()
    done = False
    step = 0

    print("Started simulation!")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    print("Simluation finished! Saved results.")
    env.close()

if __name__ == "__main__":
    run_evaluation()