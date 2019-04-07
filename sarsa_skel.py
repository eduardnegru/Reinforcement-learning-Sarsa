""" Exemplu de cod pentru tema 1.
"""

import matplotlib.pyplot as plt
import numpy as np
import gym
import math
from tqdm import tqdm
import gym_minigrid  # pylint: disable=unused-import
from argparse import ArgumentParser

def softmax(Q, state, N, c, env, args):

	beta = 0

	if (N.get(state, None) is None):
		N[state] = 0

	actions = [(a1, a2) for (i1, a1) in enumerate(env.actions) for (i2, a2) in enumerate(env.actions) if a1 != a2 and i2 > i1]

	max = None

	if N[state] > 0:
		for (a1, a2) in actions:
			differece = abs(Q.get(state, {}).get(a1, 0) - Q.get(state, {}).get(a2, 0))
			if max is None or max < differece:
				max = differece
		
		if(max == 0):
			beta = 1
		else:
			beta = math.log(N[state]) / float(max)

	# sum_qa = sum([math.exp(beta * Q.get((state, action), 0)) for action in env.actions])
	sum_qa = 0.0

	for action in env.actions:
		sum_qa += math.exp(beta * Q.get(state, {}).get(action, 0))

	probabilities = [math.exp(beta * Q.get(state, {}).get(action, 0)) / float(sum_qa) for action in env.actions]

	return np.random.choice(list(env.actions), p = probabilities)

def epsilon_greedy(Q, state, N, c, env, args):

	maxActions = []
	
	if(N.get(state, None) is None):
		N[state] = 0
	
	if(N[state] != 0):
		epsilon = float(c) / N[state]
	else:
		epsilon = 1

	actionToValue = {}

	for action in env.actions:
		actionToValue[action] = Q.get(state, {}).get(action, 0)

	maxValue = max(actionToValue.values())
	actionCount = len(actionToValue.keys())

	deltaDifference = 0.0001

	maxActions = [action for action, value in actionToValue.items() if abs(maxValue - value) < deltaDifference]

	probabilities = [(float(epsilon) / actionCount + float(1 - epsilon) / len(maxActions)) if action in maxActions else float(epsilon) / actionCount for action, value in actionToValue.items()]
	
	# print(actionToValue.keys())
	return np.random.choice(list(actionToValue.keys()), p = probabilities)

def upper_confidence_bound(Q, state, N, c, env, args, Nsa):
	
	maxAction = env.actions.left
	maxValue = None

	if N.get(state, None) is None:
		N[state] = 0

	for action in env.actions:

		rap = 0

		if Nsa.get(state, None) is None:
			Nsa[state] = {}
			Nsa[state][action] = 0
		else:
			if Nsa[state].get(action, None) is None:
				Nsa[state][action] = 0

		if Nsa[state][action] == 0:
			rap = 1
		else:
			rap = math.log(N[state]) / float(Nsa[state][action])
		
		value = Q.get(state, {}).get(action, 0) + c * math.sqrt(rap)
		
		if maxValue is None or value > maxValue:
			maxValue = value
			maxAction = action

	return maxAction

def take_action(strategy, Q, str_state, N, env, args, Nsa = {}):
	
	if strategy == "softmax":
		return softmax(Q, str_state, N, args.c, env, args)
	elif strategy == "e-greedy":
		return epsilon_greedy(Q, str_state, N, args.c, env, args)
	elif strategy == "ucb":
		return upper_confidence_bound(Q, str_state, N, args.c, env, args, Nsa)
	else:
		return env.action_space.sample()

def main():

	parser = ArgumentParser()

	# Learning rate
	parser.add_argument("--learning_rate", type=float, default=0.01,
						help="Learning rate -> Alpha")
	
	#Discount
	parser.add_argument("--discount", type=float, default=0.9,
						help="Discount -> Gamma.")

	#Epochs
	parser.add_argument("--epochs", type=int, default=50000,
		help="Number of epochs")

	#Report frequency
	parser.add_argument("--report_frequency", type=int, default=500,
		help="Result report frequency")

	parser.add_argument("--c", type=float, default=0.05,
                        help="Epsilon greedy constant.")
	#Environment type
	parser.add_argument("--environment", type=str, default="MiniGrid-Empty-6x6-v0",
		help="Environment type")

	#Strategy
	parser.add_argument("--strategy", nargs="+", default=["e-greedy"],
		help="Strategy. Possible values : softmax or e-greedy")

	#Default Q value
	parser.add_argument("--q0", type=int, default=0,
		help="Default Q value")

	#Default Q value
	parser.add_argument("--graphics", type=bool, default=False,
		help="Run the program with graphics")

	parser.add_argument("--seed", type=int, default=None,
		help="Seed to generate maps")

	#Plot image file name
	parser.add_argument("--file_name", type=str, default="file",
		help="Plot image file name")

	

	args = parser.parse_args()

	env = gym.make(args.environment)

	# renderer = env.render("human")

	steps, avg_returns, avg_lengths = {}, {}, {}
	recent_returns, recent_lengths = {}, {}

	for strategy in args.strategy:
		avg_lengths[strategy] = []
		avg_returns[strategy] = []
		steps[strategy] = []
		recent_returns[strategy] = []
		recent_lengths[strategy] = []

	for strategy in args.strategy:

		victory = 0
		lost = 0
		crt_return, crt_length = 0, 0

		#declare Q and N
		Q = {}
		N = {}
		Nsa = {}

		if(args.seed is not None):
			env.seed(args.seed)

		state, done = env.reset(), False
		str_state = str(env)

		for step in range(1, args.epochs + 1):

			# print(step)

			action = take_action(strategy, Q, str_state, N, env, args, Nsa)

			while (done == False):

				if(N.get(str_state, None) is None):
					N[str_state] = 0
				else:
					N[str_state] += 1

				if strategy == "ucb":
					if Nsa.get(str_state, None) is None:
						Nsa[str_state] = {}
						Nsa[str_state][action] = 0
					else:
						if Nsa[str_state].get(action, None) is None:
							Nsa[str_state][action] = 0
						else:
							Nsa[str_state][action] += 1

				next_state, reward, done, _ = env.step(action)

				str_next_state = str(env)

				crt_return += reward
				crt_length += 1

				next_action = take_action(strategy, Q, str_next_state, N, env, args, Nsa)

				if Q.get(str_state, None) is None:
					Q[str_state] = {}
					Q[str_state][action] = args.q0
				else:
					if Q[str_state].get(action, None) is None:
						Q[str_state][action] = args.q0

				if Q.get(str_next_state, None) is None:
					Q[str_next_state] = {}
					Q[str_next_state][next_action] = args.q0
				else:
					if Q[str_next_state].get(next_action, None) is None:
						Q[str_next_state][next_action] = args.q0

				Q[str_state][action] +=  args.learning_rate * (reward + args.discount * Q[str_next_state][next_action] - Q[str_state][action])

				state = next_state
				action = next_action
				str_state = str_next_state
				
				if(args.graphics is not False):
					env.render("human")

			#After the epoch reset the state
			if crt_return > 0:
				victory += 1
				# print("Victory")
			else:
				lost += 1
				# print("Lost")

			if(args.seed is not None):
				env.seed(args.seed)

			state, done = env.reset(), False
			recent_returns[strategy].append(crt_return)  # câștigul episodului încheiat
			recent_lengths[strategy].append(crt_length)
			crt_return, crt_length = 0, 0
			
			if step % args.report_frequency == 0:

				avg_return = np.mean(recent_returns[strategy])  # media câștigurilor recente
				avg_length = np.mean(recent_lengths[strategy])  # media lungimilor episoadelor
				steps[strategy].append(step)  # pasul la care am reținut valorile
				avg_returns[strategy].append(avg_return)
				avg_lengths[strategy].append(avg_length)

				print(  # pylint: disable=bad-continuation
					f"Step {step:4d}"
					f" | Avg. return = {avg_return:.2f}"
					f" | Avg. ep. length: {avg_length:.2f}"
					f" | Victory percentage: {victory / float(lost + victory) if lost + victory != 0 else 100:.2f}"
				)

				recent_returns[strategy].clear()
				recent_lengths[strategy].clear()

	file_name = str(args.environment) + "_" + str(args.strategy) + "_" + str(args.learning_rate) + "_" + str(args.c) + "_" + str(args.epochs)

	description = "Map=" + args.environment + " Strategy=" + str(args.strategy) + " Alpha=" + str(args.learning_rate) + " Gamma=" + str(args.discount) + " C=" + str(args.c) + " Epochs=" + str(args.epochs)
	
	
	plt.figure(figsize=(15, 5))
	plt.suptitle(description)
	plt.subplot(131)
	plt.title("Average episode length")

	for strategy in args.strategy:
		plt.plot(steps[strategy], avg_lengths[strategy], c = ("r" if strategy == "e-greedy" else ("b" if strategy == "softmax" else "g")), label=strategy)

	plt.legend()

	plt.subplot(132)
	plt.title("Average episode return")

	for strategy in args.strategy:
		plt.plot(steps[strategy], avg_returns[strategy], c = ("r" if strategy == "e-greedy" else ("b" if strategy == "softmax" else "g")), label=strategy)
	
	plt.legend()

	plt.savefig(file_name, format='pdf')
	plt.show()
		
		# _fig, (ax1, ax2) = plt.subplots(ncols=2)
		# ax1.plot(steps, avg_lengths, label=strategy)
		# plt.suptitle(description, fontsize = 7)
		
		# ax1.set_title("Average episode length")
		# ax1.legend()

		# ax2.plot(steps, avg_returns, label=strategy)
		# ax2.set_title("Average episode return")
		# ax2.legend()
		# plt.savefig(file_name, format='pdf')

if __name__ == "__main__":
	main()
