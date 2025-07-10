



### This isn't used in training. This is used to have the agent play a game
### to see what it does ( for development and testing purposes... ).

def run_dummy_agent(env, render=True):
    obs = env.reset()
    done = False

    print("-----------------------------------------------")
    print("---------Dummy agent taking their turn---------")
    print("-----------------------------------------------")

    print(f"Bot's hand: {env.player_bot.hand}")

    previous_phase = env.phase

    while not done:
        print(f"Current phase...{env.phase}")
        if env.phase in env.action_spaces:
            choice = env.action_spaces[env.phase].sample()
            print(f"Choice: phase {env.phase}, option {choice}...")
        else:
            choice = None # <--- Buy phase?
            print(f"Choice: phase {env.phase}...")

        obs, reward, done, _ = env.step(choice)

        if render:
            print(f"\nPhase summary: {previous_phase}, Reward: {reward}, Done: {done}\n")

        previous_phase = env.phase


"""
-----------------------------------------------
---------Dummy agent taking their turn---------
-----------------------------------------------
Bot's hand: 2 Copper, 3 Estate
Current phase...action
Action: 2...
Bot is passing the action phase...

Phase summary: action, Reward: 0.0, Done: False

Current phase...money
Else action = 0...
Bot is told to play all treasure cards...

Phase summary: money, Reward: 0.0, Done: False

Current phase...buy
Action: 0...                              ????????????????????????????????????????????
Bot is attempting to buy Gardens...  <--- ??? need to add code to discern between  ???
                                          ??? the action of playing an action card ???
                                          ??? and the action of buying             ???
Traceback (most recent call last):        ????????????????????????????????????????????
"""