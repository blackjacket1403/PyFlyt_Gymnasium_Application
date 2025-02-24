import gymnasium as gym
import PyFlyt.gym_envs  # Ensure PyFlyt is installed
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback

wandb.init(
    project="pyflyt-training_2",
    name="SAC_QUAD_POLE_BALANCING_3",
    sync_tensorboard=True,
    monitor_gym=False,
)

def train_model(timesteps=1000000, save_path="SAC_quad_pole_balance_3"):
    """Trains PPO on the PyFlyt QuadX-Pole-Balance environment."""
    train_env = make_vec_env("PyFlyt/QuadX-Pole-Balance-v3", n_envs=6)  
    model = SAC("MlpPolicy", train_env, verbose=1, tensorboard_log="./SAC_tensorboard/", device="cuda")
    model.learn(total_timesteps=timesteps, callback=WandbCallback())
    model.save(save_path)
    train_env.close()

def test_model(load_path="SAC_quad_pole_balance_3", num_episodes=50):
    """Tests the trained SAC model with rendering for multiple episodes."""
    model = SAC.load(load_path)
    test_env = gym.make("PyFlyt/QuadX-Pole-Balance-v3", render_mode="human") 

    for episode in range(num_episodes):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward:.2f}")

    test_env.close()

if __name__ == "__main__":
    train_model()  # Train without rendering
    test_model()   # Test for 50 episodes with rendering

    # Close WandB
    wandb.finish()
