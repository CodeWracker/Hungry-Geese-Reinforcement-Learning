from HungryGeeseEnv import *
import plotly.express as px
from tqdm import tqdm

def select_row(obs,rows):
    i = 0
    cont =0
    for a in obs:
        cont+=1
        i+=cont*a
    return i % rows
def train(learning_rate = 0.001,discount_rate = 0.8,exploration_rate = 1, max_exploration_rate = 1, min_exploration_rate = 0.01, exploration_decay_rate = 0.01,rows=257):
    env = HungryGeeseGym()
    action_space_size = env.action_space.n
    state_space_size = env.observation_space_size

    q_table = np.zeros((rows, action_space_size))
    #q_table

    num_episodes = 1000
    max_steps_per_episode = 200


    rewards_all_episodes = []
    # Q-learning algorithm
    for episode in tqdm(range(num_episodes)):
        # initialize new episode params
        state = env.reset()
        done = False
        rewards_current_episode = 0.000000000
        for step in range(max_steps_per_episode): 
            # Exploration-exploitation trade-off
            # Take new action
            # Update Q-table
            # Set new state
            # Add new reward    
            exploration_rate_threshold = random.uniform(0, 1)
            
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[select_row(state,rows),:]) 
            else:
                action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            #print(float(reward))
            # Update Q-table for Q(s,a)
            q_table[select_row(state,rows), action] = q_table[select_row(state,rows), action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * np.max(q_table[select_row(new_state,rows), :]))
            state = new_state
            rewards_current_episode = rewards_current_episode+ reward 
            if done == True: 
                break
            
                
        # Exploration rate decay
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
        rewards_all_episodes.append(rewards_current_episode)
        # Exploration rate decay   
        # Add current episode reward to total rewards list

    # Calculate and print the average reward per thousand episodes
    num_med = 10
    rewards_per_n_episodes = np.split(np.array(rewards_all_episodes),num_episodes/num_med)
    count = num_med
    reward_med = []
    try:
        os.mkdir("./data/"+str(rows)+"-"+str(learning_rate))
    except:
        q_table
    np.savetxt("./data/"+str(rows)+"-"+str(learning_rate)+"/qtable.csv", q_table, delimiter=",")

    #print("********Average reward per "+str(num_med)+" episodes********\n")
    for r in rewards_per_n_episodes:
        #print(count, ": ", str(sum(r/num_med)))
        reward_med.append(float(sum(r/num_med)))
        count += num_med

    

    # Plot cumulative reward  
    #print(reward_med)
    df = pd.DataFrame(reward_med,columns=[rows])
    df.to_csv("./data/"+str(rows)+"-"+str(learning_rate)+"/results.csv")
    #df = df[rows]
    #print(df)
    #df.rolling(window=num_med).mean().plot()
    #plt.show()


    return df