import os
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from trading_env import TradingEnvironment
from data_fetcher import BinanceDataFetcher

class TuneReportCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=1000):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = 5
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    if done or truncated:
                        break
                rewards.append(episode_reward)
            
            mean_reward = np.mean(rewards)
            net_worth = self.eval_env.net_worth
            sharpe = self._calculate_sharpe()
            
            session.report({
                "mean_reward": mean_reward,
                "net_worth": net_worth,
                "sharpe_ratio": sharpe,
                "timesteps": self.n_calls
            })
        return True
    
    def _calculate_sharpe(self):
        if len(self.eval_env.returns_history) < 2:
            return 0
        returns = np.array(self.eval_env.returns_history)
        return np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

def train_ppo(config, data, features):
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    train_features = features[:train_size]
    val_data = data[train_size:]
    val_features = features[train_size:]
    
    train_env = TradingEnvironment(train_data, train_features)
    val_env = TradingEnvironment(val_data, val_features)
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        clip_range=config["clip_range"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        policy_kwargs={
            "net_arch": [config["hidden_size"]] * config["n_layers"]
        },
        verbose=0,
        tensorboard_log=None
    )
    
    callback = TuneReportCallback(val_env, eval_freq=2048)
    model.learn(total_timesteps=config["timesteps"], callback=callback)
    
    val_env.reset()
    done = False
    while not done:
        obs = val_env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        _, _, done, _, _ = val_env.step(action)
    
    final_return = (val_env.net_worth - val_env.initial_balance) / val_env.initial_balance
    
    session.report({
        "final_return": final_return,
        "final_net_worth": val_env.net_worth,
        "total_trades": len(val_env.trades)
    })

def optimize_hyperparameters(data, features, num_samples=20, max_timesteps=50000):
    config = {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "gamma": tune.uniform(0.9, 0.999),
        "clip_range": tune.uniform(0.1, 0.3),
        "n_steps": tune.choice([256, 512, 1024, 2048]),
        "n_epochs": tune.choice([3, 5, 10]),
        "hidden_size": tune.choice([64, 128, 256]),
        "n_layers": tune.choice([2, 3, 4]),
        "timesteps": max_timesteps
    }
    
    scheduler = ASHAScheduler(
        metric="sharpe_ratio",
        mode="max",
        max_t=max_timesteps,
        grace_period=10000,
        reduction_factor=3
    )
    
    result = tune.run(
        lambda config: train_ppo(config, data, features),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": 2},
        verbose=1,
        local_dir="./ray_results"
    )
    
    best_trial = result.get_best_trial("sharpe_ratio", "max", "last")
    best_config = best_trial.config
    
    print(f"\nBest hyperparameters found:")
    for key, value in best_config.items():
        if key != "timesteps":
            print(f"  {key}: {value}")
    
    return best_config