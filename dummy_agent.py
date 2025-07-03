def run_dummy_agent(env, render=True):
    obs = env.reset()
    done = False
    while not done:
        if env.phase in env.action_spaces:
            action = env.action_spaces[env.phase].sample()
        else:
            action = 0
        obs, reward, done, _ = env.step(action)
        if render:
            print(f"Phase: {env.phase}, Reward: {reward}, Done: {done}")
